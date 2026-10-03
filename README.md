---
title: Dhaga Returns Classifier
emoji: 🧵
colorFrom: orange
colorTo: gray
sdk: docker
app_port: 7860
---

# Dhaga and Co Returns Reason Classifier

An internal Streamlit tool for Neha's weekly review: classify free-text return reasons, inspect uncertain rows, and rank vendor/category/size combinations by fit-complaint count. Python plus LangChain, two OpenRouter models, no database or separate backend.

## Run locally

Python 3.10 or newer is required. From this folder on Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env locally and add OPENROUTER_API_KEY. Do not paste your key into chat.
streamlit run app.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501). On Windows use `py -m venv .venv`, `.venv\Scripts\activate`, and `copy .env.example .env` instead. You can also enter your key in the app's password field for the current browser session. The sample preview works without a key; paid model calls do not.

This workspace already has `.venv` installed. You can launch directly with:

```bash
.venv/bin/streamlit run app.py
```

## Use the app

1. Select **Bundled synthetic sample** or upload a UTF-8 CSV.
2. Choose how many rows to process (default 25, up to 1,000), or select **Process all rows** for the entire validated CSV. Then press **Classify returns**. Existing dropdown reasons map in code; only Other comments need model classification.
3. Read the reason breakdown and fit-complaint investigation list. Counts are not return rates or causal evidence.
4. Inspect **Review** for unclear, ambiguous, invalid or failed rows; the trace retains both models' answers when available.
5. Download processed rows and inspect the run report. Ordinary reruns and downloads reuse session results. Pressing Classify again deliberately starts a new paid run.
6. Use **Try a comment** for one example. Try `theek nahi tha` as the ambiguous demo case.

Input changes clear batch results. Results are in browser-session memory only; download them before closing the session. The app processes the dates in the file, not an implicit last-week filter. The sample contains 25 deliberately varied, hand-authored synthetic rows; it is not client data or a quality benchmark.

## Expected CSV

```csv
return_id,sku_id,vendor_id,category,size_ordered,return_date,dropdown_reason,other_text
R-001,SKU-001,V-17,Kurti,M,2026-09-28,Other,"size chota hai, kamar pe tight"
```

All columns except `return_id` are required. IDs remain strings, with surrounding whitespace trimmed. Missing or repeated return IDs are accepted and preserved; the app adds a unique `row_id` within each uploaded file and a `data_quality_note`. Missing IDs, repeated IDs, and identical rows after whitespace cleaning are flagged in the cleaning summary and review queue. No rows are automatically deleted; retained duplicates still contribute to complaint counts. Tracking fields are included in the labelled export and reserved in input files. Dates must be valid YYYY-MM-DD. Blank vendor/category/size values remain visible but cannot be ranked. Extra columns are retained in export; only `other_text` reaches a model.

Accepted dropdowns (case insensitive): Other, Fit, Size issue, Too small, Too large, Quality, Poor quality, Not as shown, Wrong or damaged item, Wrong item, Damaged item, Delivery, Late delivery, Changed mind, Unclear; underscore forms of canonical labels also work where defined in `processing.py`. Unknown values stop the run so mappings can be reviewed. The actual teammate CSV is pending, so its schema may need mapping before use.

Limits: 10 MB, 10,000 rows per file; subset runs up to 1,000 rows or the entire file with Process all rows, 4,000 characters per comment. Ranked groups default to at least three eligible fit complaints. This is an adjustable demo threshold, not statistical significance. Small groups and missing metadata are counted explicitly.

Export escapes leading spreadsheet formula characters with an apostrophe. Original in-memory text remains unchanged. Do not re-upload a labelled export as raw input.

## Files and workflow

- `app.py`: three Streamlit tabs, session state and downloads.
- `classifier.py`: schemas, prompts, model calls, validation and escalation.
- `processing.py`: CSV checks, bounded batch concurrency, counts and costs.
- `evaluate.py`: optional labelled evaluation command.
- `tests/test_app.py`: offline pipeline, processing and UI checks.
- `docs/dev_spec.md`: implementation scope and acceptance criteria.
- `docs/build_note.md`: patterns, tradeoffs, cost arithmetic and verification limits.

LangChain prompt chaining connects classification and detail extraction. `RunnableBranch` selects Fit, Quality, or Not-as-shown detail extraction; other categories bypass it. Low confidence, multiple reasons, or unresolved details cause a single stronger-model pass. Every model output uses a Pydantic schema and exact-substring evidence validation. Invalid outputs get one validation retry. Transient requests get one explicit retry; the SDK's long default retry policy is disabled.

Statuses: **Classified** = trusted specific model label; **Passed through** = mapped dropdown; **Unclear** = insufficient information; **Review** = unresolved valid guess; **Unclassified** = schema/evidence validation exhausted; **Failed** = provider/internal error. Only Classified and Passed through fit rows enter rankings. Existing dropdowns receive no inferred fit direction.

## Models and cost

Defaults are `google/gemini-2.5-flash-lite` for bulk work and `google/gemini-2.5-flash` for escalation. Both use temperature 0 to reduce sampling variation; this does not guarantee identical results. The stronger-model benefit must be measured on actual examples. Reasoning is disabled with a 1,024-token output cap. Override IDs in `.env`; choose two distinct models supporting JSON schema, temperature 0 and disabled reasoning. No agent, memory, retrieval, or model training is used.

The app reads public OpenRouter catalogue token rates at run time and reports the snapshot time. For each model:

`estimated USD = reported input tokens × input USD/token + reported output tokens × output USD/token`

No rate or incomplete usage means unknown cost, never zero. All returned validation attempts are counted. Failed requests can have unreported usage. Estimates exclude account fees and can differ from provider billing or cache discounts; verify actual charges in OpenRouter activity. Optional INR conversion uses a rate you supply. The weekly projection divides measured run cost by processed Other rows and multiplies by 6,547, conditional on a representative sample and the case-study rate assumptions. No rupee saving is claimed.

Official integration references: [LangChain ChatOpenRouter](https://docs.langchain.com/oss/python/integrations/chat/openrouter), [OpenRouter structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs), [Streamlit session state](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state).

## Verification

Offline checks make no paid calls:

```bash
python -m unittest discover -s tests -v
```

After configuring a key, this evaluation **makes paid model calls**:

```bash
python evaluate.py data/eval_smoke.csv
```

It prints accuracy, coverage, per-reason results, fit-direction accuracy, token-cost estimate and runtime; predictions are saved to the ignored `evaluation_results.csv`. Failed, unclassified and review rows count as incorrect. The bundled 12 examples are smoke cases used during development, not a held-out gold set. Prepare a separate manually labelled set before claiming the proposed 90% accuracy target. No live accuracy result is currently claimed.

## Deploy

A Dockerfile and Hugging Face Space metadata are included. Create a **Docker Space**, upload this repository without `.env`, and add `OPENROUTER_API_KEY` as a Space secret (or leave it unset and use the per-session key field). `SMALL_MODEL` and `STRONG_MODEL` can be Space variables. The app listens on port 7860. Anyone able to use a publicly hosted app with a server key can consume that key's credit; for a public demo without access control, prefer session-entered keys.

The hosted URL and Docker build have not been verified in this environment. Local launch and offline verification are separate from hosted deployment. This MVP deliberately has no authentication, durable storage, real integrations or generated weekly brief.
