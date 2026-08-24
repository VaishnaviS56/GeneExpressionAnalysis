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

By default, the runner evaluates in batches of 10 and appends a checkpoint to
`evaluators/langsmith_results.txt` after each batch. To change this:

```powershell
.\.venv\Scripts\python.exe -m evaluators.run_evals single --batch-size 5 --results-file evaluators\my_results.txt
```

Use `--no-seed` when the LangSmith datasets already exist and you do not want
to update examples.

Run hallucination/groundedness evaluation plus human review queueing:

```powershell
$env:OPENAI_API_KEY = "..."
.\.venv\Scripts\python.exe -m evaluators.run_quality_evals single
.\.venv\Scripts\python.exe -m evaluators.run_quality_evals multi
.\.venv\Scripts\python.exe -m evaluators.run_quality_evals all
```

This uses `HALLUCINATION_JUDGE_MODEL`, defaulting to `gpt-5`, for
`response_groundedness` and `evidence_validity`, then adds each completed
experiment batch to the `Target Discovery Response Quality` annotation queue.
Use `--no-human-review` to run only the judge evaluators.
