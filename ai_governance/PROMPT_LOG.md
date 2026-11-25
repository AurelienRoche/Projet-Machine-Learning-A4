# Prompt log

4–6 high-stakes interactions (at least one per student), showing how the LLM was constrained with schemas, assumptions and leakage guardrails. Updated every week.

## 1. Jérémie – Week 2 – M1: plan before implementation

> Before writing any code, review the repo and the LAB 5 brief and list what the Week 2 deliverable still lacks. Constraints: no code until we agree on a plan; the test set must not influence any preprocessing choice; B1 must stay as defined in the specification. Output: a table "missing item / why it matters" and numbered steps.

**Outcome:** plan reviewed and validated before coding: shared `src/` module, B1 kept minimal, extra cleaning steps tested as cross-validation variants instead of being applied by default.

## 2. Jérémie – Week 2 – M1: leak-free preprocessing module

> Implement `src/secom.py` with `load_secom()`, `get_split()` and `make_preprocessor()`. Constraints: split before any preprocessing; every fitted step is an sklearn transformer inside a Pipeline; Fail = 1; seed 42; only the model may see y. Return the code and a check of the output shape for each variant.

**Outcome:** module used by all notebooks. Review of the output found a flaw in the correlation filter (AI_AUDIT #3).
