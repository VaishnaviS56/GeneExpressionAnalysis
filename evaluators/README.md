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

Download LangSmith traces locally:

```powershell
.\.venv\Scripts\python.exe -m evaluators.download_traces --file-prefix evaluators\judge_traces
```

By default this downloads root traces from the LangSmith `evaluators` project
and saves JSONL files in chunks of 1000:

```text
evaluators\judge_traces_evaluators_chunk_0001.jsonl
evaluators\judge_traces_evaluators_chunk_0002.jsonl
```

You can pass an API key directly or let it read `LANGSMITH_API_KEY` from `.env`:

```powershell
.\.venv\Scripts\python.exe -m evaluators.download_traces --api-key "..." --file-prefix evaluators\judge_traces
```

To download evaluation experiment projects instead of the `evaluators` project:

```powershell
.\.venv\Scripts\python.exe -m evaluators.download_traces --project-prefix target-discovery --file-prefix evaluators\eval_traces
```

Use `--include-child-runs` if you want every child run as a separate exported
record instead of only root traces.

Convert downloaded trace files to CSV:

```powershell
.\.venv\Scripts\python.exe -m evaluators.traces_to_csv evaluators\eval_traces_evaluators_chunk_0001.jsonl
```

This creates a CSV with the same name:

```text
evaluators\eval_traces_evaluators_chunk_0001.csv
```

The CSV includes `test_id`, `run_id`, evaluator name/key, score, comment,
experiment, and trace ids.

Convert local txt result files to CSV:

```powershell
.\.venv\Scripts\python.exe -m evaluators.traces_to_csv evaluators\my_results.txt --batch-size 1
.\.venv\Scripts\python.exe -m evaluators.traces_to_csv evaluators\quality_single_results.txt --batch-size 5
```

For txt files, `--batch-size` means the batch size used when the eval results
were saved, such as 1 or 5. The converter writes the CSV beside the txt file
with the same base name.

Download the human-review annotation queue as a review-friendly CSV:

```powershell
.\.venv\Scripts\python.exe -m evaluators.download_annotation_queue_excel --format csv --output evaluators\annotation_queue_review.csv
```

The CSV flattens each queued run into `test_id`, `user_query`, `final_answer`,
run metadata, expected tools, error text, and the LangSmith trace URL so it can
be given to human reviewers or passed to a separate LLM-as-judge workflow.

If the annotation queue has already been downloaded into workbook parts, convert
those local `.xlsx` files directly without calling LangSmith:

```powershell
.\.venv\Scripts\python.exe -m evaluators.download_annotation_queue_excel --format csv --from-xlsx evaluators\annotation_queue_part_001.xlsx evaluators\annotation_queue_part_002.xlsx evaluators\annotation_queue_part_003.xlsx evaluators\annotation_queue_part_004.xlsx evaluators\annotation_queue_part_005.xlsx evaluators\annotation_queue_part_006.xlsx evaluators\annotation_queue_part_007.xlsx evaluators\annotation_queue_part_008.xlsx --output evaluators\annotation_queue_review.csv
```
