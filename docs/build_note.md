# Returns Reason Classifier Build Note

Updated 3 October 2026. Local MVP implemented; real-data quality evaluation and hosted deployment remain pending.

## What was built

A Streamlit app with Analyze returns, Review, and Try a comment tabs. CSV input becomes row-level labels and evidence, a fit-complaint investigation list, a download, and a run report. The three application modules are `app.py`, `classifier.py`, and `processing.py`. No database or separate frontend service is needed. The 25-row bundled sample is explicitly synthetic; the teammate's dataset has not arrived.

## Code and model responsibilities

| Step | Responsible component | Why |
| --- | --- | --- |
| Input checks and dropdown mapping | Python | Validation and lookup need no language judgment. |
| Triage of Other comments | Gemini 2.5 Flash Lite | Interpret informal and mixed-language comments at the lower token rate. |
| Select detail schema | LangChain RunnableBranch | Route on a known label deterministically. |
| Fit, quality or mismatch extraction | Flash Lite | Map customer wording to typed attributes with evidence. |
| Schema and evidence checks | Pydantic and Python | Reject malformed output and quotes absent from the original text. |
| Difficult-row escalation | Gemini 2.5 Flash | A second, higher-priced model is reserved for uncertain or multiple observations. Its quality advantage remains to be measured. |
| Counts, ranking and exports | Python | Arithmetic, ordering and CSV creation are deterministic. |

Prompt chaining separates reason selection from detailed extraction. Without it, a single output must cover all reason-specific details at once. Routing limits detail extraction to the relevant schema; without it, irrelevant detail fields add complexity and opportunities for unsupported values. Four worker threads improve throughput but are not claimed as a separate reasoning pattern. Both models use temperature 0; that reduces variation without promising determinism. Reasoning is explicitly disabled and output capped at 1,024 tokens for this short extraction workflow.

## Failure policy

One schema/evidence retry, one transient-request retry and at most one escalation pass per row. Persistent ambiguity is visible, never silently promoted. Unclear, Review, Unclassified and Failed rows remain outside the investigation ranking. Failure on one row does not discard the batch. Existing dropdown labels pass through with no guessed fit direction. Minimum group size is an adjustable demonstration threshold, not statistical confidence.

## Cost arithmetic

The public OpenRouter catalogue checked on 3 October listed Flash Lite at $0.10 input/$0.40 output per million tokens and Flash at $0.30/$2.50. The app fetches rates again for each run rather than assuming these remain fixed.

For each model, `USD = input tokens × input rate + output tokens × output rate`. Reported usage from rejected validation attempts is included. Missing usage or unavailable pricing produces unknown cost. A rupee conversion requires the operator's exchange rate. The conditional weekly estimate is `run USD / processed Other rows × 6,547`; the volume assumes compatible per-order return percentages. No live run cost or business saving has been measured yet.

## What broke and what was verified

The installed OpenRouter integration uses **milliseconds** for timeout and a retry time budget rather than a strict retry count. Passing a conventional seconds value would have made requests time out almost immediately; relying on SDK defaults could instead wait too long. The app supplies an SDK client with a 45,000 ms timeout, disables SDK retries, and implements one bounded retry in Python.

18 offline checks passed: classification routes, escalation, evidence rejection, conflicting observations, provider failures, CSV checks, aggregate denominators, formula-safe export, evaluation scoring, Streamlit session behavior, and a real LangChain/SDK round trip over a simulated HTTP transport. Dependency consistency passed. The local Streamlit health endpoint returned `ok`. No real model call was made because a key is not configured. AppTest verifies UI behavior; a visual browser review has not been performed. Docker/public URL deployment is prepared but unverified.

## Next evidence

Configure the API key locally, run the smoke examples, adapt to the teammate CSV, then evaluate separately labelled unseen examples. Accuracy targets remain proposals. Get a working public URL and have a teammate operate the app before the 09:30 IST rehearsal. Later work includes correction persistence and intervention tracking; the MVP makes no claim that complaint counts are return rates or proven fixes.

References: [LangChain OpenRouter integration](https://docs.langchain.com/oss/python/integrations/chat/openrouter), [OpenRouter model catalogue](https://openrouter.ai/models), [reasoning token controls](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens). Case-study business figures are cited in the discovery note.
