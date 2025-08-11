import os
import io
import hashlib
import time
from typing import List, Dict, Any
import re

import streamlit as st
import pandas as pd
import plotly.express as px

from settings import (
    DEFAULT_TOP_K, MAX_UPLOAD_FILES, MAX_FILE_SIZE_MB, MAX_SNIPPET_CHARS
)
from resume_parser import extract_text_from_file, guess_name_from_filename
from pinecone_helper import (
    validate_index, upsert_text_records, search_and_rerank_text, wipe_namespace, now_ms
)
from summarizer import summarize_many

# ---------------- Pricing envs (optional display only) ----------------
EMBED_PRICE_PER_1K  = float(os.getenv("PINECONE_EMBED_PRICE_PER_1K", "0"))
RERANK_PRICE_PER_1K = float(os.getenv("PINECONE_RERANK_PRICE_PER_1K", "0"))

# ---------------- Page ----------------
st.set_page_config(page_title="Candi-Comm Engine", page_icon="🧭", layout="wide")
st.title("🧭 Candi-Comm: A Candidate Matching Engine with AI-based summaries")
st.write("Paste the Job Description, upload/paste resumes, set Top K (sidebar), then click **Run for matches**.")

# --------- Namespace (auto per session) ----------
def _auto_run_namespace(seed: str = "") -> str:
    t = str(int(time.time()))
    h = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8] if seed else ""
    return f"run-{t}-{h}" if h else f"run-{t}"

if "run_namespace" not in st.session_state:
    st.session_state.run_namespace = _auto_run_namespace()

# ---- persistence helpers ----
if "last_run" not in st.session_state:
    st.session_state.last_run = None  # stores rows/summaries/analytics and CSV bytes

def _build_table(rows: List[Dict[str, Any]], summaries: List[str]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Rank": i + 1,
        "Name/ID": r.get("name", f"Candidate {i+1}"),
        "Base Score": round(float(r.get("score", 0.0)), 6),
        "Rerank Score": round(float(r.get("rerank_score", r.get("score", 0.0))), 6),
        "Summary": summaries[i] if i < len(summaries) else ""
    } for i, r in enumerate(rows)])

def stash_last_run(*, rows, summaries, upserted_chars, openai_spend_usd, openai_tokens_in, openai_tokens_out, query_chars):
    df = _build_table(rows, summaries)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.session_state.last_run = {
        "rows": rows,
        "summaries": summaries,
        "df": df,
        "csv": csv_bytes,
        "upserted_chars": int(upserted_chars),
        "openai_spend_usd": float(openai_spend_usd),
        "openai_tokens_in": int(openai_tokens_in),
        "openai_tokens_out": int(openai_tokens_out),
        "query_chars": int(query_chars),
    }

# --------- Sidebar ----------
with st.sidebar:
    st.subheader("Settings")

    k = st.slider(
        "Top K", 3, 95, value=min(max(DEFAULT_TOP_K, 3), 95), step=1,
        help="ℹ️ Number of candidates to return from Pinecone."
    )
    clean_before = st.toggle(
        "Clean namespace before ingest", value=False,
        help="ℹ️ Deletes existing data in this run namespace before ingest."
    )
    max_snip = st.slider(
        "Max snippet characters", 500, 8000, MAX_SNIPPET_CHARS, 100,
        help="ℹ️ Controls how much resume text is displayed."
    )

    # Quiet connection check; show only status
    try:
        _ = validate_index()
        pinecone_status = "🟢 Pinecone: Connected"
    except Exception:
        pinecone_status = "🔴 Pinecone: Not connected"
    openai_status = "🟢 OpenAI: Ready" if OPENAI_API_KEY else "🟠 OpenAI: Not configured"
    st.markdown(f"**Status**\n\n- {pinecone_status}\n- {openai_status}")

    st.caption("Actions")
    if st.button("Hard delete ALL in this namespace", key="btn_wipe_ns",
                 help="ℹ️ Permanently deletes all records in the current run namespace."):
        deleted, err = wipe_namespace(namespace=st.session_state.run_namespace)
        if err:
            st.error(f"Delete error: {err}")
        else:
            st.success(f"Wiped namespace '{st.session_state.run_namespace}' (deleted ~{deleted} records).")

# --------- Inputs ----------
jd = st.text_area("Job Description", height=220, placeholder="Paste the JD here…")

uploads = st.file_uploader(
    "Upload candidate resumes (PDF/DOCX/TXT)",
    type=["pdf", "docx", "txt"], accept_multiple_files=True,
    help="ℹ️ Batch upsert (≤96). Pinecone will embed, then we search & re-rank."
)

pasted_resumes = st.text_area(
    "Or paste resumes (one per candidate)",
    value="",
    height=200,
    help=(
        "ℹ️ Use a line starting with `### Name` for each resume, e.g.\n"
        "### Jane Doe\n<her resume text>\n### John Smith\n<his resume text>\n"
        "…or separate blocks with a line of ---"
    )
)

# Single action button
run_all = st.button("Run for matches", key="btn_run_all", type="primary", use_container_width=True)

# --------- Helpers ----------
def _parse_pasted_resumes(raw: str) -> List[Dict[str, Any]]:
    raw = (raw or "").strip()
    if not raw:
        return []

    blocks: List[Dict[str, Any]] = []

    headers = list(re.finditer(r"(?m)^###\s*(.+?)\s*$", raw))
    if headers:
        for i, h in enumerate(headers):
            name = h.group(1).strip() or f"Pasted{i+1}"
            start = h.end()
            end = headers[i+1].start() if i + 1 < len(headers) else len(raw)
            text = raw[start:end].strip()
            if text:
                blocks.append({"name": name, "text": text})
        return blocks

    parts = re.split(r"(?m)^\s*(?:---+|\*\*\*+|===+)\s*$", raw)
    for i, part in enumerate(parts, start=1):
        text = part.strip()
        if text:
            blocks.append({"name": f"Pasted{i}", "text": text})
    return blocks

def build_records_from_pasted(raw: str) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    t = now_ms()
    for i, blk in enumerate(_parse_pasted_resumes(raw), start=1):
        recs.append({
            "id": f"resume::pasted::{t}::{i}",
            "text": blk["text"],
            "name": blk["name"],
            "source_filename": f"pasted_{i}.txt",
            "ingested_at": t
        })
    return recs

def build_records_from_uploads(files) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    t = now_ms()
    for f in files[:MAX_UPLOAD_FILES]:
        if f.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.warning(f"Skipping {f.name}: file > {MAX_FILE_SIZE_MB} MB.")
            continue
        txt = extract_text_from_file(f.name, f.read())
        if not txt.strip():
            st.warning(f"Skipping {f.name}: parsed text is empty.")
            continue
        recs.append({
            "id": f"resume::{t}::{f.name}",            # REQUIRED: 'id'
            "text": txt,                                # REQUIRED: top-level 'text'
            "name": guess_name_from_filename(f.name),   # extra searchable fields
            "source_filename": f.name,
            "ingested_at": t
        })
    return recs

def build_records_combined(files, pasted_text: str) -> List[Dict[str, Any]]:
    recs = []
    if files:
        recs.extend(build_records_from_uploads(files))
    if pasted_text and pasted_text.strip():
        recs.extend(build_records_from_pasted(pasted_text))
    return recs

def render_results(rows: List[Dict[str, Any]] = None, summaries: List[str] = None):
    # Recover from state if not provided (lets this function run independently)
    if rows is None or summaries is None:
        if not st.session_state.last_run:
            st.info("No results to show yet.")
            return
        rows = st.session_state.last_run["rows"]
        summaries = st.session_state.last_run["summaries"]
        df = st.session_state.last_run["df"]
        csv_bytes = st.session_state.last_run["csv"]
    else:
        df = _build_table(rows, summaries)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        # keep stash in sync so analytics/download survive reruns
        st.session_state.last_run = (st.session_state.last_run or {})
        st.session_state.last_run.update({
            "rows": rows, "summaries": summaries, "df": df, "csv": csv_bytes
        })

    st.subheader("Top Matches (Re-ranked)")

    for i, r in enumerate(rows):
        base = float(r.get("score", 0.0))
        rerank = float(r.get("rerank_score", base))
        c1, c2, c3 = st.columns([5, 2, 2])
        with c1:
            st.markdown(f"**{i+1}. {r.get('name','—')}**")
        with c2:
            st.markdown(f"Base: `{base:.4f}`")
        with c3:
            st.markdown(f"Rerank: `{rerank:.4f}`")

        st.markdown(summaries[i] if i < len(summaries) else "")

        with st.expander("Resume snippet"):
            st.text_area(" ", value=(r.get("text") or "")[:max_snip], height=180, key=f"snip_{i}")

        st.markdown("---")

    # Download CSV — uses prebuilt bytes so no heavy work happens on click
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="matches.csv",
        mime="text/csv",
        key="btn_dl_csv"
    )

def render_analytics(rows: List[Dict[str, Any]] = None, *,
                     upserted_chars: int = None,
                     openai_spend_usd: float = None,
                     openai_tokens_in: int = None,
                     openai_tokens_out: int = None,
                     query_chars: int = None):
    # If not given, recover every input from the stash so this function stands alone
    if rows is None:
        if not st.session_state.last_run:
            st.info("No analytics to show yet.")
            return
        rows = st.session_state.last_run["rows"]
        upserted_chars = st.session_state.last_run["upserted_chars"]
        openai_spend_usd = st.session_state.last_run["openai_spend_usd"]
        openai_tokens_in = st.session_state.last_run["openai_tokens_in"]
        openai_tokens_out = st.session_state.last_run["openai_tokens_out"]
        query_chars = st.session_state.last_run["query_chars"]

    st.subheader("📊 Run Analytics")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Candidates returned", len(rows))
    with col2:
        st.metric("Upserted text (chars)", upserted_chars)
    with col3:
        st.metric("Query length (chars)", query_chars)

    # Histogram
    base_scores = [float(r.get("score", 0.0)) for r in rows]
    st.plotly_chart(
        px.histogram(x=base_scores, nbins=10, title="Base Vector Score Distribution"),
        use_container_width=True
    )

    # Costs
    st.markdown("### 💸 Cost (this run)")
    embed_tokens_est = upserted_chars // 4
    rerank_tokens_est = (query_chars // 4) * len(rows) + sum(len((r.get("text") or "")) // 4 for r in rows)

    pinecone_embed_cost = (embed_tokens_est / 1000.0) * EMBED_PRICE_PER_1K if EMBED_PRICE_PER_1K > 0 else None
    pinecone_rerank_cost = (rerank_tokens_est / 1000.0) * RERANK_PRICE_PER_1K if RERANK_PRICE_PER_1K > 0 else None

    cost_cols = st.columns(3)
    with cost_cols[0]:
        st.metric("OpenAI summaries (USD)", f"${openai_spend_usd:.4f}")
        st.caption(f"prompt={openai_tokens_in} tok, completion={openai_tokens_out} tok")
    with cost_cols[1]:
        st.metric("Pinecone embed (est.)", "—" if pinecone_embed_cost is None else f"${pinecone_embed_cost:.4f}")
        if EMBED_PRICE_PER_1K == 0:
            st.caption("Set PINECONE_EMBED_PRICE_PER_1K to show $")
    with cost_cols[2]:
        st.metric("Pinecone rerank (est.)", "—" if pinecone_rerank_cost is None else f"${pinecone_rerank_cost:.4f}")
        if RERANK_PRICE_PER_1K == 0:
            st.caption("Set PINECONE_RERANK_PRICE_PER_1K to show $")

def search_with_retry(jd_text: str, k: int, attempts: int = 6, delay: float = 1.0) -> List[Dict[str, Any]]:
    last: List[Dict[str, Any]] = []
    for i in range(1, attempts + 1):
        rows = search_and_rerank_text(
            query_text=jd_text,
            namespace=st.session_state.run_namespace,
            top_k=k,
            fields=["name", "text", "source_filename", "ingested_at"]
        )
        if rows:
            return rows
        time.sleep(delay * i)  # 1s, 2s, 3s...
        last = rows
    return last

# --------- Pipeline ---------
def run_pipeline():
    if not jd.strip():
        st.error("Please paste a Job Description.")
        return

    # Steps: (optional clean) + build + upsert + search + summarize
    steps_total = 5 + (1 if clean_before else 0)
    progress = st.progress(0)
    steps_done = 0

    with st.status("Running…", expanded=True) as status:
        # Step 0 — optional clean
        if clean_before:
            status.write("🗑️ Cleaning namespace before ingest…")
            deleted, err = wipe_namespace(namespace=st.session_state.run_namespace)
            if err:
                status.update(label="Pre-clean failed", state="error")
                st.error(f"Namespace clean error: {err}")
                return
            status.write("✅ Cleaned namespace")
            steps_done += 1; progress.progress(min(100, int(steps_done/steps_total*100)))

        # Step 1 — Build records
        status.write("📥 Reading resumes…")
        records = build_records_combined(uploads, pasted_resumes)
        n_resumes = len(records)

        # Guardrail: if we're ingesting new resumes, enforce K <= count
        if n_resumes > 0 and k > n_resumes:
            status.update(label="Top K is too large", state="error")
            st.warning(f"Top K ({k}) is greater than the number of resumes provided this run ({n_resumes}). "
                       f"Please set Top K to ≤ {n_resumes} and try again.")
            return

        upserted_chars = sum(len(r["text"]) for r in records)
        status.write(f"Parsed {len(records)} resumes.")
        steps_done += 1; progress.progress(min(100, int(steps_done/steps_total*100)))

        # Step 2 — Upsert (Pinecone embeds)
        if records:
            status.write("⬆️ Upserting text records to Pinecone (batch ≤96)…")
            try:
                count = upsert_text_records(records, namespace=st.session_state.run_namespace)
                status.write(f"✅ Upserted {count} records.")
            except Exception as e:
                status.update(label="Upsert failed", state="error")
                st.error(f"Upsert error: {e}")
                return
        else:
            status.write("ℹ️ No new resumes to upsert; searching existing content in this namespace.")
        steps_done += 1; progress.progress(min(100, int(steps_done/steps_total*100)))

        # Step 3 — Search + rerank (server-side)
        status.write("🔎 Searching with query text and re-ranking…")
        rows = search_with_retry(jd, k, attempts=6, delay=1.0)
        if not rows:
            status.update(label="Finished — no results", state="error")
            st.warning("No results returned. If you just ingested, wait a few seconds and click Run again.")
            return
        status.write(f"✅ Retrieved {len(rows)} results.")
        steps_done += 1; progress.progress(min(100, int(steps_done/steps_total*100)))

        # Step 4 — Summaries (OpenAI)
        status.write("🧠 Generating summaries…")
        summaries, usage = summarize_many(job_desc=jd, items=rows, max_in_chars=1600, max_out_tokens=120)
        status.write("✅ Summaries ready.")
        steps_done += 1; progress.progress(min(100, int(steps_done/steps_total*100)))

        # Persist everything for reruns (download, reload)
        stash_last_run(
            rows=rows,
            summaries=summaries,
            upserted_chars=upserted_chars,
            openai_spend_usd=usage["spent_usd"],
            openai_tokens_in=usage["prompt_tokens"],
            openai_tokens_out=usage["completion_tokens"],
            query_chars=len(jd or "")
        )

        # Ensure progress bar hits 100% and status completes
        progress.progress(100)
        status.update(label="Done ✅", state="complete")

    # Render now (and they’ll persist for reruns)
    render_results()
    render_analytics()

# --------- Entry ---------
if run_all:
    run_pipeline()
elif st.session_state.last_run:
    # Rehydrate UI from previous run if the user reloads or clicks Download
    render_results()
    render_analytics()
