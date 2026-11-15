// Routing: which free model serves one request, and what the router learns
// from refusals.
//
// Every decision is a function of the brief and the gateway answers seen so
// far, never of the clock, so a replay takes the same path. Candidates are
// ordered by the brief's rating_milli first (quality is 60% of the score),
// then by fewest calls so far (least-used tie-break, deterministic), then by
// id. A model that refuses three calls in a row is exhausted for the rest of
// the trial; a 402 exhausts it at once. The env sends one prompt at a time,
// so no per-model queue is needed here.

import {
  capabilitiesOf,
  meetsRequirements,
  type ModelRow,
  type Requirement,
} from './providers.ts';

/** How many refusals in a row mark a model exhausted (a 402 does it at once). */
export const REFUSALS_TO_EXHAUST = 3;
/** The most gateway attempts spent on one prompt (well under the 60-call cap). */
export const MAX_ATTEMPTS_PER_PROMPT = 6;

/** One model's trial state: what the router has learned about it. */
export interface ModelState {
  model: ModelRow;
  /** Gateway calls made to this model, refusals included. */
  calls: number;
  /** Refusals since its last success. */
  consecutiveRefusals: number;
  /** True once the model is known spent. */
  exhausted: boolean;
}

/** The router over one trial's free models. */
export class Router {
  readonly states: ModelState[];

  constructor(models: readonly ModelRow[]) {
    this.states = models
      .filter((m) => m.free)
      .map((model) => ({ model, calls: 0, consecutiveRefusals: 0, exhausted: false }));
  }

  /** The free models meeting `requires`, best rating first. */
  candidatesFor(requires: Requirement): ModelState[] {
    const out = this.states.filter(
      (s) => !s.exhausted && meetsRequirements(requires, capabilitiesOf(s.model)),
    );
    out.sort((a, b) => {
      if (b.model.rating_milli !== a.model.rating_milli) return b.model.rating_milli - a.model.rating_milli;
      if (a.calls !== b.calls) return a.calls - b.calls;
      return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
    });
    return out;
  }

  /** A call answered 2xx with a parseable body: the model is healthy again. */
  noteSuccess(state: ModelState): void {
    state.calls += 1;
    state.consecutiveRefusals = 0;
  }

  /** A call refused with `status` (502 for a network error): learn from it. */
  noteRefusal(state: ModelState, status: number): void {
    state.calls += 1;
    if (status === 402) {
      state.exhausted = true;
      return;
    }
    state.consecutiveRefusals += 1;
    if (state.consecutiveRefusals >= REFUSALS_TO_EXHAUST) state.exhausted = true;
  }
}
