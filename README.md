# GEA Agent

Chat UI + agent that classifies a user query as:
- **general**: answer normally
- **technical (genes)**: extract genes, pull a high-confidence (**0.700**) Homo sapiens (**9606**) interaction network from **STRING**, and build a weighted **NetworkX** graph

## Notes
- STRING calls require outbound network access at runtime.
- The agent expects gene symbols (e.g., `TP53`, `EGFR`, `BRCA1`) in the query for “technical”.

