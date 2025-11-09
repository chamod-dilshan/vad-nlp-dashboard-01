import streamlit as st
import pandas as pd
import plotly.express as px
from utils.loader import load_all_models
from utils.inference import predict_single_label, predict_multilabel_topk

# ------------------------------------------------------------
# Streamlit page setup & style
# ------------------------------------------------------------
st.set_page_config(page_title="VAD NLP Insights", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
.small-note {color:#6b7280; font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

st.title("🫀 Extracting Safety Insights from VAD Reports (NLP Dashboard)")
st.caption("Analyze sentiment, severity, and topic signals from VAD adverse-event narratives.")

# ------------------------------------------------------------
# Color maps and order (low → high)
# ------------------------------------------------------------
SENTIMENT_ORDER = ["Non-negative/mild", "Negative", "Strongly Negative"]
SEVERITY_ORDER  = ["Mild/Moderate", "Serious", "Fatal"]

SENTIMENT_COLORS = {
    "Non-negative/mild": "#22c55e",  # green
    "Negative":          "#3b82f6",  # blue
    "Strongly Negative": "#991b1b",  # dark red
}
SEVERITY_COLORS = {
    "Mild/Moderate": "#22c55e",
    "Serious":       "#3b82f6",
    "Fatal":         "#991b1b",
}

def enforce_order(df, col, order):
    if col in df.columns:
        df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    return df

# ------------------------------------------------------------
# Load all models
# ------------------------------------------------------------
(models_sent, models_sev, models_top) = load_all_models()
tok_s, mdl_s, meta_s = models_sent
tok_v, mdl_v, meta_v = models_sev
tok_t, mdl_t, meta_t = models_top

def map_id2label(meta, idx: int):
    id2label = meta.get("id2label", {})
    if not id2label:
        return str(idx)
    return id2label.get(str(idx), id2label.get(idx, str(idx)))

# ------------------------------------------------------------
# File upload and processing
# ------------------------------------------------------------
st.sidebar.header("Upload file")
uploaded_file = st.sidebar.file_uploader("Upload a file with FOI_TEXT column", type=["xlsx", "csv"])

if uploaded_file:
    try:
        if uploaded_file.name.lower().endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            try:
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding="latin-1")
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    st.success(f"Loaded {len(df)} rows.")

    if "FOI_TEXT" not in df.columns:
        st.error("Missing column FOI_TEXT.")
        st.stop()

    texts = df["FOI_TEXT"].fillna("").astype(str).tolist()
    results = []

    for text in texts:
        s_pred, s_probs     = predict_single_label(tok_s, mdl_s, text)
        sev_pred, sev_probs = predict_single_label(tok_v, mdl_v, text)
        top_preds, top_scores = predict_multilabel_topk(
            tok_t, mdl_t, text, meta_t["label_space"], meta_t["topic_labels"]
        )
        top_topic_names = [t[1] for t in top_preds]
        results.append({
            "Text": text[:300] + ("..." if len(text) > 300 else ""),
            "Sentiment": map_id2label(meta_s, s_pred),
            "Severity":  map_id2label(meta_v, sev_pred),
            "Top Topics": ", ".join(top_topic_names),
            "TopTopicList": top_topic_names
        })

    res_df = pd.DataFrame(results)
    res_df = enforce_order(res_df, "Sentiment", SENTIMENT_ORDER)
    res_df = enforce_order(res_df, "Severity",  SEVERITY_ORDER)

    # --------------------------------------------------------
    # KPI summary
    # --------------------------------------------------------
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Rows analyzed", len(res_df))
    with c2: st.metric("Unique topic labels (top-k)", len(set([t for row in res_df["TopTopicList"] for t in row])))
    with c3: st.markdown('<div class="small-note">Models: Sentiment/Severity + Multilabel Topics (top-k)</div>', unsafe_allow_html=True)

    # --------------------------------------------------------
    # Data table
    # --------------------------------------------------------
    st.subheader("Preview")
    st.dataframe(res_df.drop(columns=["TopTopicList"]), use_container_width=True, hide_index=True)

    # --------------------------------------------------------
    # Distributions
    # --------------------------------------------------------
    st.subheader("Distributions")
    d1, d2 = st.columns(2)

    with d1:
        st.markdown("**Sentiment**")
        fig = px.histogram(
            res_df, x="Sentiment", color="Sentiment",
            category_orders={"Sentiment": SENTIMENT_ORDER},
            color_discrete_map=SENTIMENT_COLORS
        )
        fig.update_layout(xaxis_title="", yaxis_title="Count", bargap=0.15)
        st.plotly_chart(fig, use_container_width=True)

    with d2:
        st.markdown("**Severity**")
        fig2 = px.histogram(
            res_df, x="Severity", color="Severity",
            category_orders={"Severity": SEVERITY_ORDER},
            color_discrete_map=SEVERITY_COLORS
        )
        fig2.update_layout(xaxis_title="", yaxis_title="Count", bargap=0.15)
        st.plotly_chart(fig2, use_container_width=True)

    # --------------------------------------------------------
    # Topic distribution (from top-k lists)
    # --------------------------------------------------------
    st.subheader("Topic Distribution (Top-k votes)")
    topics_long = (
        res_df[["TopTopicList"]]
        .explode("TopTopicList")
        .rename(columns={"TopTopicList": "Topic"})
        .dropna()
    )
    topic_counts = (
        topics_long.value_counts("Topic")
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )

    QUAL = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel + px.colors.qualitative.Bold
    fig3 = px.bar(
        topic_counts, x="Topic", y="Count",
        text="Count",
        color="Topic",
        color_discrete_sequence=QUAL
    )
    fig3.update_traces(textposition="outside")
    fig3.update_layout(xaxis_title="", yaxis_title="Count",
                       uniformtext_minsize=10, uniformtext_mode="hide")
    st.plotly_chart(fig3, use_container_width=True)

else:
    st.info("Upload a dataset to start analysis.")
