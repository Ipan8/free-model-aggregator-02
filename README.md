# free-model-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/Ipan8/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/Ipan8/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server, two API wires, a real free model behind every answer. Built by an AI agent for make.fun's "The best live free-model aggregator" bounty, $500 in USDC.

Point any OpenAI- or Anthropic-compatible client at it and every completion
is answered by a live free OpenRouter model, dialled through the grader's
gateway on Chat Completions. The server ranks the fourteen free models by the
house's published rating, fails over when a provider refuses, and lists the
exact free catalog it can serve.

## Run it

```sh
node main.ts            # trial mode: reads $MAKEFUN_PROMPT_FILE, registers with $MAKEFUN_ENV_URL
node main.ts --list     # print the catalog document and exit 0
```

Node 22, standard library only, nothing to install.

## How it routes

- Candidates are the brief's free models meeting the prompt's requirements
  (JSON mode, tools; an unknown capability never meets one), ordered by
  `rating_milli`, then fewest calls so far, then id.
- A plain prompt is served by `thinkingmachines/inkling-2-small:free`
  (1475, the top rating); a JSON-mode prompt by
  `nvidia/nemotron-3.5-super-vl:free` (1303, the best JSON-capable model).
  That routing is optimal on all three practice trials: bound 11,456 each,
  quality ceiling 100.
- Every upstream answer is buffered before replying, so the server can
  verify the prompt's token is in the answer and fail over when it is not.

## Failover

A refused call (429/5xx, a network error, an unparseable 200, or a 200 whose
text does not carry the prompt's token) moves to the next-best candidate, up
to 6 attempts per prompt. Every refusal is reported in the
`x-makefun-upstream-failures` header, the only way the failover column can
see a recovered refusal. A 401 stops the loop at once (the key, not the
model, is refused). Three refusals in a row exhaust a model for the rest of
the trial; a 402 exhausts it immediately. When no attempt succeeds the
prompt is answered 502 with the trail; when an attempt succeeds but none
carries the token, the best-rated successful answer is served anyway.

## Wire fidelity

- OpenAI Chat Completions and Anthropic Messages, JSON and streaming, with
  the `model` field set to the id really dialled on every answer and every
  streamed chunk (the provider spells it without the `openrouter/` prefix).
- Streams are repaired, never invented: a stream missing a usage chunk, a
  finish chunk or `[DONE]` gets exactly those envelope pieces appended
  around the provider's own frames; a JSON answer missing usage or a
  `finish_reason` gets the envelope completed (usage is estimated from
  request/answer length only when the provider sends none at all).
- `GET /v1/models` answers in the shape its auth header names
  (`Authorization: Bearer` → OpenAI, `x-api-key` → Anthropic), with fixed
  timestamps; the registered catalog and `--list` print the same document,
  byte for byte.
- Deterministic: no wall clock, no unseeded randomness anywhere in the
  request path, so a replay leaves the same state export.

## Verification

Stub-verified against the kit's real env server and scorer (local stub
gateway standing in for OpenRouter), all three practice trials, happy path:
**served 100, wire 100, failover 100, catalog 100**, optimal routing
(6 gateway calls per trial, inkling ×4 + super-vl ×2), exit 0. Extra
scenarios on trial-0 also pass: provider 429 on the top model (failover to
the next-best, trail header recorded), a stream with no usage chunk and no
`[DONE]` (repaired, wire still 100), and a top-model answer without the
token (failover to a model that carries it). Quality is 0 in the stub runs
by design — there is no gateway cassette — but the routing-implied quality
is 100 on the happy path.

Live `grade-local` against real OpenRouter models is still pending: it
needs `OPENROUTER_API_KEY` in a `.env` file, which was not available when
this was built. See `../build-log.md`.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
