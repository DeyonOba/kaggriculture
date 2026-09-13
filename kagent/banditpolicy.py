"""Play-time side of the single-deviation counterfactual bandit.

`rules` (rl/bandit.py fit) holds one ridge model per (head, offset): the
predicted paired-margin gain of applying that offset to the champion plan in
a given state (168-dim features). Each day the provider scores every model
and applies the best offset when its predicted gain clears `min_gain` —
one deviation per day, matching the regime the data was collected in.
Pure Python (no numpy) so it runs inside the Kaggle bundle.
"""

from kagent import features, planner


class BanditPolicy:
    def __init__(self, rules, max_per_day=1):
        self.rules = rules
        self.models = rules.get("models", [])
        self.min_gain = float(rules.get("min_gain", 1500.0))
        self.max_per_day = max_per_day
        self.last_decision = None  # (day, [(head, offset_value, predicted_gain)])
        # Head lookup is built lazily: in a flat bundle the `features` alias
        # is bound only after every module has executed, and this object is
        # constructed at module level.
        self._head_index = None

    def _predict(self, model, x):
        mu, sd, w = model["mu"], model["sd"], model["w"]
        total = model["b"]
        for i in range(len(w)):
            total += w[i] * ((x[i] - mu[i]) / sd[i])
        return total

    def make_plan(self, view, memory):
        plan = planner.make_day_plan(view, memory, speculative=True)
        if not self.models:
            return plan
        if self._head_index is None:
            self._head_index = {name: i for i, (name, _) in enumerate(features.HEADS)}
        x = features.encode(view, memory)
        head_masks = features.masks(view)
        scored = []
        for m in self.models:
            h = self._head_index.get(m["head"])
            if h is None or not head_masks[h][m["off_idx"]]:
                continue
            gain = self._predict(m, x)
            if gain >= self.min_gain:
                scored.append((gain, h, m["off_idx"]))
        if not scored:
            self.last_decision = (view.day, [])
            return plan
        scored.sort(reverse=True)
        choices = features.residual_choices(plan, view, head_masks)
        applied = []
        used_heads = set()
        for gain, h, off in scored:
            if h in used_heads or len(applied) >= self.max_per_day:
                continue
            choices[h] = off
            used_heads.add(h)
            applied.append((features.HEADS[h][0], features.HEADS[h][1][off], round(gain)))
        self.last_decision = (view.day, applied)
        return features.apply_residual(plan, choices, view)


class KnobBanditPolicy:
    """Season-long knob version: at KNOB_DAY (shops revealed) score every
    (knob, value) arm with its fitted gain model and set the best ones on
    `memory.knobs` for the rest of the game; before that day, and when no
    arm clears `min_gain`, the plan is exactly champ-h3."""

    KNOB_DAY = 6
    DECISION_DAYS = (6, 9, 12)  # rules may carry "day"; conditions may test consumers or price ratios

    def __init__(self, rules, max_knobs=None):
        self.models = rules.get("models", [])
        self.min_gain = float(rules.get("min_gain", 1500.0))
        self.max_knobs = max_knobs if max_knobs is not None else int(rules.get("max_knobs", 1))
        self.conditions = rules.get("conditions", [])
        self.last_decision = None

    def _predict(self, model, x):
        mu, sd, w = model["mu"], model["sd"], model["w"]
        total = model["b"]
        for i in range(len(w)):
            total += w[i] * ((x[i] - mu[i]) / sd[i])
        return total

    def make_plan(self, view, memory):
        decided = getattr(memory, "knobs_decided", None)
        if not isinstance(decided, set):
            decided = set() if not decided else {self.KNOB_DAY}
            memory.knobs_decided = decided
        if view.day in self.DECISION_DAYS and view.day not in decided:
            decided.add(view.day)
            chosen = dict(getattr(memory, "knobs", None) or {})
            # Explicit, interpretable rules: {"knob", "value", optional "day"
            # (default 6), "min_consumers"/"max_consumers": {product: k},
            # "max_price"/"min_price": {product: ratio-to-base}} evaluated on
            # features.context at the rule's decision day.
            ctx = features.context(view)
            for rule in self.conditions:
                if int(rule.get("day", self.KNOB_DAY)) != view.day:
                    continue
                ok = all(ctx["consumers"].get(p, 0) >= k for p, k in rule.get("min_consumers", {}).items())
                ok = ok and all(ctx["consumers"].get(p, 0) <= k for p, k in rule.get("max_consumers", {}).items())
                ok = ok and all(ctx["price_ratio"].get(p, 1.0) <= r for p, r in rule.get("max_price", {}).items())
                ok = ok and all(ctx["price_ratio"].get(p, 1.0) >= r for p, r in rule.get("min_price", {}).items())
                if ok:
                    chosen[rule["knob"]] = rule["value"]
            scored = []
            if self.models:
                x = features.encode(view, memory)
                scored = sorted(
                    ((self._predict(m, x), m["head"], m["offset"]) for m in self.models), reverse=True
                )
                for gain, knob, value in scored:
                    if gain < self.min_gain or knob in chosen or len(chosen) >= self.max_knobs:
                        continue
                    chosen[knob] = value
            memory.knobs = chosen
            self.last_decision = (view.day, chosen, [(round(g), k, v) for g, k, v in scored[:3]])
        return planner.make_day_plan(view, memory, speculative=True)
