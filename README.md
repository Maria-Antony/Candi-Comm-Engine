# Candi-Comm Engine

A lightweight app to rank candidate resumes against a job description. It uses Pinecone Integrated Inference to create embeddings inside Pinecone (no local embedding code), performs semantic search, then re-ranks with Pinecone’s Inference API and generates concise fit summaries with OpenAI.

### Highlights

1. One-click pipeline: Ingest → Embed (Pinecone) → Search → Re-rank → Summarize → Visualize
2. Batch upserts (≤96) per Pinecone’s integrated inference limits
3. Same model for query & docs (handled by your Pinecone index)
4. Server-side re-rank (e.g., pinecone-rerank-v0 or your configured model)
5. Top-K slider (3–95) + guardrail if K > resumes provided in the run
6. Live status + progress while running (no “silent waiting”)
7. Results view: clean line-by-line display + Download CSV
8. OpenAI summaries with strict budget guard (default ≤ $1)

### Stack

1. Frontend: Streamlit, Plotly (analytics)
2. Search & Rerank: Pinecone (Integrated Inference + Inference Rerank API)
3. Summaries: OpenAI Chat Completions
4. Parsing: pdfplumber, python-docx, UTF-8 text
5. Config: .env via python-dotenv

### Architecture

```plaintext

[User] 
  └─> Streamlit UI
        ├─ Upload files / paste resumes
        ├─ Paste Job Description
        └─ Set Top-K, cleanup toggles
             │
             ▼
      [app.py] pipeline
        1) Build text records (id, text, name, source, ingested_at)
        2) Pinecone Index.upsert(records=...)      <-- Pinecone embeds (batch ≤96)
        3) Pinecone Index.search(inputs.text=JD, top_k=K)
        4) Pinecone Inference.rerank(model=..., query, documents)
        5) OpenAI summaries for top results (budget-guarded)
        6) Visualize + CSV export + analytics
             │
             ▼
        [Streamlit session state]
        keeps last results visible across reruns (e.g., CSV clicks)
```

### How to use

1. Paste the Job Description.
2. Upload resumes (PDF/DOCX/TXT) or paste multiple resumes separated by ### Name sections or --- lines.
3. Choose Top-K (3–95) from the sidebar.
4. Click Run. You’ll see live steps: Ingest → Embed → Search → Re-rank → Summarize.
5. View results line-by-line, open per-candidate snippets, and Download CSV.
6. The app warns and stops if Top-K exceeds the number of resumes ingested in this run.
7. If you’re querying an existing namespace, Pinecone may return fewer than K if content is limited.
