# GEA Agent LangSmith Evaluations

Set LangSmith credentials before running:

```powershell
$env:LANGSMITH_API_KEY = "..."
$env:LANGSMITH_TRACING = "true"
```

The first run creates or updates these LangSmith datasets from the text files:

- `TargetDiscovery-SingleTurn` from `singleQs.txt`
- `TargetDiscovery-MultiTurn` from `MultiTurnQs.txt`

Reload datasets without running evaluations:

```powershell
.\.venv\Scripts\python.exe -m evaluators.load_datasets single
.\.venv\Scripts\python.exe -m evaluators.load_datasets multi
.\.venv\Scripts\python.exe -m evaluators.load_datasets all
```

If you removed or renamed question IDs locally, add `--prune-removed` so the
LangSmith dataset exactly matches the text files:

```powershell
.\.venv\Scripts\python.exe -m evaluators.load_datasets all --prune-removed
```

Run evaluations:

```powershell
.\.venv\Scripts\python.exe -m evaluators.run_evals single
.\.venv\Scripts\python.exe -m evaluators.run_evals multi
.\.venv\Scripts\python.exe -m evaluators.run_evals all
```

Use Gemma 4 26B through Google AI Studio for the agent:

```powershell
$env:GOOGLE_API_KEY = "..."
$env:LLM_PROVIDER = "gemma"
$env:GEMMA_MODEL = "gemma-4-26b-a4b-it"
$env:GEMMA_TEMPERATURE = "1"
$env:GEMMA_TOP_K = "64"
$env:GEMMA_TOP_P = "0.95"
$env:GEMMA_THINKING_LEVEL = "high"
$env:LLM_RATE_LIMIT_RETRIES = "6"
$env:LLM_RATE_LIMIT_BACKOFF_SECONDS = "20"
```

The screenshot's `context length` is the model capacity. Set
`GEMMA_MAX_TOKENS` only when you want to cap generated output length. For
Ollama's local `gemma4:26b` tag, use `LLM_PROVIDER=ollama` and set
`OLLAMA_MODEL=gemma4:26b` instead.

Gemma 4 26B on Google AI Studio can have a low per-minute input-token quota.
For long eval runs, keep concurrency low and batch small:

```powershell
.\.venv\Scripts\python.exe -m evaluators.run_evals single --max-concurrency 0 --batch-size 1
```

By default, the runner evaluates in batches of 10 and appends a checkpoint to
`evaluators/langsmith_results.txt` after each batch. To change this:

```powershell
.\.venv\Scripts\python.exe -m evaluators.run_evals single --batch-size 5 --results-file evaluators\my_results.txt
```

Run one test case or resume from a specific case:

```powershell
.\.venv\Scripts\python.exe -m evaluators.run_evals single --case ST-007
.\.venv\Scripts\python.exe -m evaluators.run_evals multi --from-case MT-012
```

Use `--no-seed` when the LangSmith datasets already exist and you do not want
to update examples.

Run hallucination/groundedness evaluation plus human review queueing:

```powershell
$env:GOOGLE_API_KEY = "..."
$env:HALLUCINATION_JUDGE_MODEL = "gemini-3.5-flash"
.\.venv\Scripts\python.exe -m evaluators.run_quality_evals single
.\.venv\Scripts\python.exe -m evaluators.run_quality_evals multi
.\.venv\Scripts\python.exe -m evaluators.run_quality_evals all
```

Quality runs support the same case filters:

```powershell
.\.venv\Scripts\python.exe -m evaluators.run_quality_evals single --case ST-007
.\.venv\Scripts\python.exe -m evaluators.run_quality_evals all --from-case MT-012
```

This uses `HALLUCINATION_JUDGE_MODEL`, defaulting to `gemini-3.5-flash`, for
`response_groundedness` and `evidence_validity`, then adds each completed
experiment batch to the `Target Discovery Response Quality` annotation queue.
Use `--no-human-review` to run only the judge evaluators.
