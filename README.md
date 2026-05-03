# GEA Agent (LangGraph + Streamlit)

Chat UI + agent that classifies a user query as:
- **general**: answer normally
- **technical (genes)**: extract genes, pull a high-confidence (**0.700**) Homo sapiens (**9606**) interaction network from **STRING**, and build a weighted **NetworkX** graph

## Quickstart

1) Create env + install deps:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

2) Set LLM credentials (OpenAI-compatible):

```powershell
$env:GROQ_API_KEY="..."
```

3) Run the UI:

```powershell
streamlit run ui\\app.py
```

## Notes
- STRING calls require outbound network access at runtime.
- The agent expects gene symbols (e.g., `TP53`, `EGFR`, `BRCA1`) in the query for “technical”.

