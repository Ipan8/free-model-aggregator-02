// free-model-aggregator-02: one server, two client wires, a real free model
// behind every answer.
//
// The brief at $MAKEFUN_PROMPT_FILE lists fourteen free OpenRouter models with
// the house's ratings and the six prompts' manifest. This server:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port", "catalog"};
//   3. serves GET /v1/models (OpenAI shape for Authorization: Bearer,
//      Anthropic shape for x-api-key), POST /v1/chat/completions and
//      POST /v1/messages, JSON or server-sent events;
//   4. answers each prompt by dialling a free model through the gateway's Chat
//      Completions route (an Anthropic request is translated there and back);
//   5. polls GET $MAKEFUN_ENV_URL/work and exits 0 when it says done.
//
// ROUTING: candidates are the brief's free models meeting the prompt's
// requirements (JSON mode, tools; an unknown capability never meets one),
// ordered by rating_milli, then fewest calls, then id — so a plain prompt is
// served by the best-rated free model and a JSON-mode prompt by the best
// JSON-capable one. Requirements come from the brief's manifest row for the
// arriving prompt, unioned with what the request body itself declares.
//
// FAILOVER: a refused call (429/5xx, a network error, a 200 with an
// unparseable body, or a 200 whose text does not carry the prompt's token)
// moves to the next-best candidate, up to MAX_ATTEMPTS_PER_PROMPT attempts.
// Every refusal is reported in the x-makefun-upstream-failures header, the
// only way the failover column can see a recovered refusal. A 401 stops the
// loop at once (the key, not the model, is the problem). When no attempt
// succeeds the prompt is answered 502 with the trail; when an attempt
// succeeds but none carries the token, the best-rated successful answer is
// served anyway (its wire checks still count).
//
// DETERMINISM: every choice is a function of the brief, the requests and the
// gateway's answers. No wall clock, no unseeded randomness: model lists use
// fixed timestamps, the only waiting is a fixed per-prompt deadline that
// bounds a hung dial, and upstream answers are buffered before replying so a
// retry never leaves a half-sent stream behind.
//
// THE MODEL FIELD: the provider's own answer spells the model without the
// gateway's `openrouter/` prefix; every answer and every streamed chunk gets
// the dialled id back, so the reported model is the model really dialled.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  freeModels,
  isRecord,
  readBrief,
  type CatalogReport,
  type Json,
  type Requirement,
} from './lib/providers.ts';
import { MAX_ATTEMPTS_PER_PROMPT, Router } from './lib/router.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  completeOpenAIStream,
  ensureOpenAIJson,
  estimateTokens,
  openAIError,
  openAIJsonText,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamText,
  openAIStreamWithModel,
  openAIToAnthropic,
  streamOfText,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The whole attempt loop for one prompt is bounded by this, inside the env's 30 s cap. */
const PROMPT_DEADLINE_MS = 28_000;
/** The header that reports recovered refusals to the failover column. */
const FAILURES_HEADER = 'x-makefun-upstream-failures';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
const router = new Router(freeModels(brief));
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/** One refusal, as the trail header reports it. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

/** One served answer, buffered whole: the model that dialled it and its body. */
interface Served {
  model: string;
  stream: boolean;
  json: Json | null;
  sse: string | null;
  text: string;
}

interface ServeOutcome {
  served: Served | null;
  failures: FailureRow[];
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, unioned with the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  const fromBody = requirementsFromBody(body);
  if (entry === undefined) return fromBody;
  return {
    ...(entry.requires.tools === true || fromBody.tools === true ? { tools: true } : {}),
    ...(entry.requires.vision === true || fromBody.vision === true ? { vision: true } : {}),
    ...(entry.requires.json === true || fromBody.json === true ? { json: true } : {}),
    ...(entry.requires.min_context !== undefined ? { min_context: entry.requires.min_context } : {}),
  };
}

/** Requirements read off a request body. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const hasTools =
    (Array.isArray(body.tools) && body.tools.length > 0) ||
    messages.some(
      (m) =>
        isRecord(m) &&
        Array.isArray(m.content) &&
        m.content.some((b) => isRecord(b) && (b.type === 'tool_use' || b.type === 'tool_result')),
    );
  return {
    tools: hasTools,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/**
 * The prompt token the answer must carry, read off the request body: the
 * brief's manifest publishes no token, and each body names exactly one
 * (e.g. ZEBRA-4721) in its instruction. Null when the body names none.
 */
function tokenFor(body: Json): string | null {
  const text = JSON.stringify(body.messages ?? body);
  const match = /[A-Z]{2,}-[0-9]{4}/.exec(text);
  return match === null ? null : match[0];
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, extraHeaders: Record<string, string> = {}): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), extraHeaders);
  else sendJson(res, status, openAIError(message, type, null), extraHeaders);
}

function trailHeaders(failures: readonly FailureRow[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { [FAILURES_HEADER]: JSON.stringify(failures.slice(0, 8)) };
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The gateway error code of a refused call's body, or a plain http_<status>. */
function errorCodeOf(text: string, status: number): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      if (typeof value.error.code === 'string' && value.error.code !== '') return value.error.code.slice(0, 64);
      if (typeof value.error.type === 'string' && value.error.type !== '') return value.error.type.slice(0, 64);
    }
  } catch {
    // Not JSON: fall through to the status.
  }
  return `http_${status}`;
}

/** One dial of one model: the gateway's status and raw body text. Throws on a network error or abort. */
async function dial(model: string, body: Json, signal: AbortSignal): Promise<{ status: number; text: string }> {
  const res = await fetch(chatCompletionsUrl(), {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      authorization: `Bearer ${GATEWAY_KEY}`,
      ...(body.stream === true ? { accept: 'text/event-stream' } : {}),
    },
    body: JSON.stringify({ ...body, model }),
    signal,
  });
  return { status: res.status, text: await res.text() };
}

/** An OpenAI-shaped client body as the upstream Chat Completions body for `model`. */
function openAIUpstreamBody(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  if (typeof out.max_tokens === 'number') out.max_tokens = Math.min(Math.max(1, Math.floor(out.max_tokens)), 8192);
  else if (typeof out.max_completion_tokens === 'number') out.max_completion_tokens = Math.min(Math.max(1, Math.floor(out.max_completion_tokens)), 8192);
  else out.max_tokens = 1024;
  if (out.stream === true) out.stream_options = { include_usage: true };
  else delete out.stream_options;
  return out;
}

/**
 * Serve one prompt through the gateway: try the candidates in rating order,
 * failing over on every refusal and on an answer that does not carry the
 * prompt's token, and keep the best-rated successful answer as the fallback.
 */
async function serveThroughGateway(
  upstreamBody: Json,
  requires: Requirement,
  expectedToken: string | null,
  signal: AbortSignal,
): Promise<ServeOutcome> {
  const candidates = router.candidatesFor(requires).slice(0, MAX_ATTEMPTS_PER_PROMPT);
  const failures: FailureRow[] = [];
  let fallback: Served | null = null;
  const requestChars = JSON.stringify(upstreamBody).length;
  const estInput = estimateTokens(requestChars);

  for (const state of candidates) {
    if (signal.aborted) break;
    const model = state.model.id;
    const body: Json = { ...upstreamBody, model };
    let answer: { status: number; text: string };
    try {
      answer = await dial(model, body, signal);
    } catch {
      if (signal.aborted) break;
      failures.push({ status: 502, code: 'upstream_unreachable', model });
      router.noteRefusal(state, 502);
      continue;
    }
    if (answer.status === 401) {
      // The key, not the model, is refused: no other model can serve this prompt.
      failures.push({ status: 401, code: errorCodeOf(answer.text, 401), model });
      router.noteRefusal(state, 401);
      break;
    }
    if (answer.status < 200 || answer.status >= 300) {
      failures.push({ status: answer.status, code: errorCodeOf(answer.text, answer.status), model });
      router.noteRefusal(state, answer.status);
      continue;
    }
    const stream = body.stream === true;
    if (stream) {
      const text = openAIStreamText(answer.text);
      const served: Served = {
        model,
        stream: true,
        json: null,
        sse: completeOpenAIStream(answer.text, model, estInput, estimateTokens(text.length)),
        text,
      };
      router.noteSuccess(state);
      if (expectedToken === null || text.includes(expectedToken)) return { served, failures };
      if (fallback === null) fallback = served;
      continue;
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(answer.text);
    } catch {
      failures.push({ status: 502, code: 'bad_upstream_json', model });
      router.noteRefusal(state, 502);
      continue;
    }
    if (!isRecord(parsed)) {
      failures.push({ status: 502, code: 'bad_upstream_json', model });
      router.noteRefusal(state, 502);
      continue;
    }
    const json = ensureOpenAIJson(parsed, estInput, estimateTokens(openAIJsonText(parsed).length));
    const text = openAIJsonText(json);
    const served: Served = { model, stream: false, json, sse: null, text };
    router.noteSuccess(state);
    if (expectedToken === null || text.includes(expectedToken)) return { served, failures };
    if (fallback === null) fallback = served;
  }
  return { served: fallback, failures };
}

/** Reply with a served answer in the OpenAI wire. */
async function replyOpenAI(res: ServerResponse, served: Served, failures: FailureRow[]): Promise<void> {
  const headers = trailHeaders(failures);
  if (!served.stream) {
    sendJson(res, 200, openAIJsonWithModel(served.json ?? {}, served.model), headers);
    return;
  }
  res.writeHead(200, {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
    ...headers,
  });
  try {
    for await (const frame of openAIStreamWithModel(streamOfText(served.sse ?? ''), served.model)) res.write(frame);
  } finally {
    res.end();
  }
}

/** Reply with a served answer in the Anthropic wire. */
async function replyAnthropic(res: ServerResponse, served: Served, failures: FailureRow[]): Promise<void> {
  const headers = trailHeaders(failures);
  if (!served.stream) {
    sendJson(res, 200, openAIToAnthropic(served.json ?? {}, served.model), headers);
    return;
  }
  res.writeHead(200, {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
    ...headers,
  });
  try {
    for await (const event of anthropicEventsFromOpenAI(streamOfText(served.sse ?? ''), served.model)) res.write(event);
  } finally {
    res.end();
  }
}

/** A prompt this server gives up on, in the client's wire, with the refusal trail. */
function giveUp(res: ServerResponse, wire: 'openai' | 'anthropic', failures: FailureRow[]): void {
  const detail = failures.length === 0 ? 'no free model in the brief meets this request' : `the gateway refused every model tried (${failures.map((f) => `${f.model ?? '?'}:${f.status}`).join(', ')})`;
  sendError(res, wire, 502, wire === 'anthropic' ? 'api_error' : 'upstream_error', detail, trailHeaders(failures));
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const expectedToken = tokenFor(body);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), PROMPT_DEADLINE_MS);
  try {
    const outcome = await serveThroughGateway(openAIUpstreamBody(body, 'auto'), requires, expectedToken, controller.signal);
    if (outcome.served === null) return giveUp(res, 'openai', outcome.failures);
    await replyOpenAI(res, outcome.served, outcome.failures);
  } finally {
    clearTimeout(timer);
  }
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const expectedToken = tokenFor(body);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), PROMPT_DEADLINE_MS);
  try {
    const translated = anthropicToOpenAI(body, 'auto');
    const outcome = await serveThroughGateway(openAIUpstreamBody(translated, 'auto'), requires, expectedToken, controller.signal);
    if (outcome.served === null) return giveUp(res, 'anthropic', outcome.failures);
    await replyAnthropic(res, outcome.served, outcome.failures);
  } finally {
    clearTimeout(timer);
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
