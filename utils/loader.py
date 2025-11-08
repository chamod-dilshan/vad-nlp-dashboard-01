import json, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st

@st.cache_resource
def load_hf_bundle(model_dir: str):
    """
    Load a single exported model bundle from /models/<name>
    Expects meta.json with at least: {"hf_subdir": "hf"}
    """
    p = Path(model_dir)
    meta_path = p / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {p}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    hf_subdir = meta.get("hf_subdir", "hf")
    hf_dir = p / hf_subdir
    if not hf_dir.exists():
        raise FileNotFoundError(f"HF directory missing: {hf_dir}")

    tok = AutoTokenizer.from_pretrained(str(hf_dir), use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(str(hf_dir))
    mdl.eval()
    if torch.cuda.is_available():
        mdl = mdl.to("cuda")
    return tok, mdl, meta

@st.cache_resource
def load_all_models(base_dir="models"):
    base = Path(base_dir)
    sentiment = load_hf_bundle(base / "sentiment_ckpt")
    severity  = load_hf_bundle(base / "severity_ckpt")
    topics    = load_hf_bundle(base / "topic_ckpt")  # <-- fixed: singular
    return sentiment, severity, topics
