# settings.py — Streamlit Cloud: read ONLY these 4 from secrets; everything else is fixed constants.
import streamlit as st

# ---------- Secrets (4 keys only) ----------
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
PINECONE_INDEX   = st.secrets.get("PINECONE_INDEX", "")
OPENAI_API_KEY   = st.secrets.get("OPENAI_API_KEY", "")
PINECONE_HOST    = st.secrets.get("PINECONE_HOST", None)

# ---------- Fixed settings (no env, no os) ----------
# Pinecone
DEFAULT_NAMESPACE       = "__default__"
PINECONE_RERANK_MODEL   = "pinecone-rerank-v0"
INTEGRATED_BATCH_LIMIT  = 96
RETENTION_DAYS          = 10

# OpenAI
OPENAI_MODEL            = "gpt-4o-mini"
MAX_OPENAI_BUDGET_USD   = 0.5
OPENAI_PRICE_IN_PER_1K  = 0.005
OPENAI_PRICE_OUT_PER_1K = 0.015

# App knobs
DEFAULT_TOP_K                 = 10
MAX_UPLOAD_FILES              = 95
MAX_FILE_SIZE_MB              = 8
MAX_SNIPPET_CHARS             = 4000
SHOW_BASELINE_COMPARE_DEFAULT = True

# import os
# # from dotenv import load_dotenv

# # Load .env once, early
# # load_dotenv()

# def _get_bool(name: str, default: bool) -> bool:
#     v = os.getenv(name)
#     if v is None:
#         return default
#     return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

# # ---------- Pinecone ----------
# # PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
# # PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "")
# # PINECONE_HOST    = os.getenv("PINECONE_HOST", None)
# DEFAULT_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "__default__")

# PINECONE_RERANK_MODEL   = os.getenv("PINECONE_RERANK_MODEL", "pinecone-rerank-v0")
# INTEGRATED_BATCH_LIMIT  = int(os.getenv("INTEGRATED_BATCH_LIMIT", "96"))
# RETENTION_DAYS          = int(os.getenv("RETENTION_DAYS", "10"))

# # ---------- OpenAI ----------
# # OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
# OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# MAX_OPENAI_BUDGET_USD   = float(os.getenv("MAX_OPENAI_BUDGET_USD", "0.5"))
# OPENAI_PRICE_IN_PER_1K  = float(os.getenv("OPENAI_PRICE_IN_PER_1K", "0.005"))
# OPENAI_PRICE_OUT_PER_1K = float(os.getenv("OPENAI_PRICE_OUT_PER_1K", "0.015"))

# # ---------- App knobs ----------
# DEFAULT_TOP_K      = int(os.getenv("DEFAULT_TOP_K", "10"))
# MAX_UPLOAD_FILES   = int(os.getenv("MAX_UPLOAD_FILES", "95"))
# MAX_FILE_SIZE_MB   = int(os.getenv("MAX_FILE_SIZE_MB", "8"))
# MAX_SNIPPET_CHARS  = int(os.getenv("MAX_SNIPPET_CHARS", "4000"))
# SHOW_BASELINE_COMPARE_DEFAULT = _get_bool("SHOW_BASELINE_COMPARE_DEFAULT", True)
