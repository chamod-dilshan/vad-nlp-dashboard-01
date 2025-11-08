import json, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st

@st.cache_resource
def load_hf_bundle(model_dir: str):
    """
    Load a single exported model bundle from /models/<name>
    """
    p = Path(model_dir)
    with open(p / "meta.json") as f:
        meta = json.load(f)
    hf_dir = p / meta["hf_subdir"]
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
    topics    = load_hf_bundle(base / "topics_ckpt")
    return sentiment, severity, topics
