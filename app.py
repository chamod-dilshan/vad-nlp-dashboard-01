# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.loader import load_all_models
from utils.inference import predict_single_label, predict_multilabel_topk

# ------------------------------------------------------------
# Page setup & lightweight theming
# ------------------------------------------------------------
st.set_page_config(page_title="VAD NLP Insights", page_icon="🫀", layout="wide")

# Theme toggle (affects CSS + plotly)
mode = st.sidebar.toggle("🌗 Dark mode", value=False)

BG_GRAD_LIGHT = """
background: radial-gradient(1200px 600px at 0% 0%, #f1f5f9 0%, rgba(241,245,249,0) 60%),
            radial-gradient(1000px 500px at 100% 0%, #eef2ff 0%, rgba(238,242,255,0) 55%);
"""
BG_GRAD_DARK = """
background: radial-gradient(1200px 600px at 0% 0%, #0b1220 0%, rgba(11,18,32,0) 60%),
            radial-gradient(1000px 500px at 100% 0%, #0f172a 0%, rgba(15,23,42,0) 55%);
"""

st.markdown(f"""
<style>
.block-container {{padding-top: 1rem; padding-bottom: 2rem;}}
.reportview-container .main .block-container {{ {BG_GRAD_DARK if mode else BG_GRAD_LIGHT} }}
section[data-testid="stSidebar"] {{ backdrop-filter: blur(6px); }}
.small-note {{ color:{('#9ca3af' if not mode else '#94a3b8')}; font-size:0.9rem; }}
.badge {{
  display:inline-block; padding:2px 8px; border-radius:999px; font-size:0.85rem; 
  background:rgba(0,0,0,0.06); margin-right:6px; border:1px solid rgba(0,0,0,0.08);
}}
.dataframe tbody tr:hover {{ background-color:{('#f8fafc' if not mode else '#0b1220')}; }}
</style>
""", unsafe_allow_html=True)

# Plotly defaults
px.defaults.template = "plotly_dark" if mode else "simple_white"
px.defaults.width = 900
px.defaults.height = 420

st.title("🫀 Extracting Safety Insights from VAD Reports")
st.caption("Analyze **sentiment**, **severity**, and **topics** from adverse-event narratives. Upload a CSV/XLSX with a **FOI_TEXT** column.")

# ------------------------------------------------------------
# Category order & colors (low → high)
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
QUAL = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel + px.colors.qualitative.Bold

def enforce_order(df, col, order):
    if col in df.columns:
        df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    return df

def map_id2label(meta, idx: int):
    id2label = meta.get("id2label", {})
    if not id2label:
        return str(idx)
    return id2label.get(str(idx), id2label.get(idx, str(idx)))

# ------------------------------------------------------------
# Load models
# ------------------------------------------------------------
(models_sent, models_sev, models_top) = load_all_models()
tok_s, mdl_s, meta_s = models_sent
tok_v, mdl_v, meta_v = models_sev
tok_t, mdl_t, meta_t = models_top

# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
st.sidebar.header("1) Upload")
uploaded_file = st.sidebar.file_uploader("File with FOI_TEXT", type=["xlsx", "csv"])

st.sidebar.header("2) Display")
preview_len = st.sidebar.slider("Text preview length", 120, 1000, 300, step=20)
show_conf = st.sidebar.checkbox("Show confidence in record viewer", value=True)

# ------------------------------------------------------------
# Main flow
# ------------------------------------------------------------
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

    if "FOI_TEXT" not in df.columns:
        st.error("Missing column **FOI_TEXT**.")
        st.stop()

    texts = df["FOI_TEXT"].fillna("").astype(str).tolist()

    rows = []
    for text in texts:
        s_pred, s_probs       = predict_single_label(tok_s, mdl_s, text)
        sev_pred, sev_probs   = predict_single_label(tok_v, mdl_v, text)
        top_preds, top_scores = predict_multilabel_topk(
            tok_t, mdl_t, text, meta_t["label_space"], meta_t["topic_labels"]
        )
        top_topic_codes = [t[0] for t in top_preds]
        top_topic_names = [t[1] for t in top_preds]

        rows.append({
            "Text": text[:preview_len] + ("..." if len(text) > preview_len else ""),
            "FullText": text,
            "Sentiment": map_id2label(meta_s, s_pred),
            "Severity":  map_id2label(meta_v, sev_pred),
            "Top Topics": ", ".join(top_topic_names),
            "TopTopicList": top_topic_names,
            "SentimentProbs": s_probs.tolist(),
            "SeverityProbs": sev_probs.tolist(),
            "TopTopicCodes": top_topic_codes
        })

    res_df = pd.DataFrame(rows)
    res_df = enforce_order(res_df, "Sentiment", SENTIMENT_ORDER)
    res_df = enforce_order(res_df, "Severity",  SEVERITY_ORDER)

    # Tabs
    tab_overview, tab_dist, tab_topics, tab_records = st.tabs(["Overview", "Distributions", "Topics", "Records"])

    # --------------------------------------------------------
    # Overview
    # --------------------------------------------------------
    with tab_overview:
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Rows analyzed", len(res_df))
        with c2: st.metric("Unique topics (top-k)", len(set([t for row in res_df["TopTopicList"] for t in row])))
        with c3:
            s_counts = res_df["Sentiment"].value_counts().reindex(SENTIMENT_ORDER).fillna(0).astype(int)
            st.metric("Most common sentiment", s_counts.idxmax() if s_counts.sum() else "—")
        with c4:
            v_counts = res_df["Severity"].value_counts().reindex(SEVERITY_ORDER).fillna(0).astype(int)
            st.metric("Most common severity", v_counts.idxmax() if v_counts.sum() else "—")

        st.markdown("**Legend**")
        st.markdown(
            f"""<span class="badge" style="border-color:{SENTIMENT_COLORS['Non-negative/mild']};color:{SENTIMENT_COLORS['Non-negative/mild']}">Non-negative/Mild</span>
                <span class="badge" style="border-color:{SENTIMENT_COLORS['Negative']};color:{SENTIMENT_COLORS['Negative']}">Negative</span>
                <span class="badge" style="border-color:{SENTIMENT_COLORS['Strongly Negative']};color:{SENTIMENT_COLORS['Strongly Negative']}">Strongly Negative</span>
                <span class="badge" style="border-color:{SEVERITY_COLORS['Mild/Moderate']};color:{SEVERITY_COLORS['Mild/Moderate']}">Mild/Moderate</span>
                <span class="badge" style="border-color:{SEVERITY_COLORS['Serious']};color:{SEVERITY_COLORS['Serious']}">Serious</span>
                <span class="badge" style="border-color:{SEVERITY_COLORS['Fatal']};color:{SEVERITY_COLORS['Fatal']}">Fatal</span>""",
            unsafe_allow_html=True
        )

        st.subheader("Preview")
        st.dataframe(
            res_df[["Text", "Sentiment", "Severity", "Top Topics"]],
            use_container_width=True, hide_index=True
        )

        # Download
        csv_bytes = res_df.drop(columns=["TopTopicList","SentimentProbs","SeverityProbs","TopTopicCodes"]).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download results (CSV)", data=csv_bytes, file_name="vad_predictions.csv", mime="text/csv")

        st.markdown('<div class="small-note">Tip: adjust preview length from the sidebar.</div>', unsafe_allow_html=True)

    # --------------------------------------------------------
    # Distributions
    # --------------------------------------------------------
    with tab_dist:
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Sentiment distribution**")
            fig = px.histogram(
                res_df, x="Sentiment", color="Sentiment",
                category_orders={"Sentiment": SENTIMENT_ORDER},
                color_discrete_map=SENTIMENT_COLORS
            )
            fig.update_layout(xaxis_title="", yaxis_title="Count", bargap=0.15)
            st.plotly_chart(fig, use_container_width=True)

        with d2:
            st.markdown("**Severity distribution**")
            fig2 = px.histogram(
                res_df, x="Severity", color="Severity",
                category_orders={"Severity": SEVERITY_ORDER},
                color_discrete_map=SEVERITY_COLORS
            )
            fig2.update_layout(xaxis_title="", yaxis_title="Count", bargap=0.15)
            st.plotly_chart(fig2, use_container_width=True)

        # Optional: severity x sentiment as a heatmap
        st.markdown("**Sentiment × Severity (heatmap)**")
        ctab = pd.crosstab(res_df["Sentiment"], res_df["Severity"]).reindex(index=SENTIMENT_ORDER, columns=SEVERITY_ORDER).fillna(0)
        fig_hm = px.imshow(ctab, text_auto=True, aspect="auto", color_continuous_scale="Blues")
        fig_hm.update_layout(xaxis_title="", yaxis_title="")
        st.plotly_chart(fig_hm, use_container_width=True)

    # --------------------------------------------------------
    # Topics
    # --------------------------------------------------------
    with tab_topics:
        st.markdown("**Topic distribution (Top-k votes)**")
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
        fig3 = px.bar(
            topic_counts, x="Topic", y="Count",
            text="Count", color="Topic",
            color_discrete_sequence=QUAL, height=500
        )
        fig3.update_traces(textposition="outside")
        fig3.update_layout(xaxis_title="", yaxis_title="Count",
                           uniformtext_minsize=10, uniformtext_mode="hide")
        st.plotly_chart(fig3, use_container_width=True)

        # Topic vs Severity (stacked)
        st.markdown("**Topic × Severity (stacked)**")
        topics_join = topics_long.join(res_df["Severity"])
        topic_sev = topics_join.groupby(["Topic","Severity"]).size().reset_index(name="Count")
        # enforce order for stacked bars
        topic_sev["Severity"] = pd.Categorical(topic_sev["Severity"], SEVERITY_ORDER, ordered=True)
        fig4 = px.bar(
            topic_sev, x="Topic", y="Count", color="Severity",
            color_discrete_map=SEVERITY_COLORS, barmode="stack", height=520
        )
        fig4.update_layout(xaxis_title="", yaxis_title="Count")
        st.plotly_chart(fig4, use_container_width=True)

    # --------------------------------------------------------
    # Records (detail viewer)
    # --------------------------------------------------------
    with tab_records:
        st.subheader("Record inspector")
        if len(res_df) == 0:
            st.info("No rows to inspect.")
        else:
            idx = st.number_input("Row index", min_value=0, max_value=len(res_df)-1, value=0, step=1)
            row = res_df.iloc[int(idx)]
            st.markdown("**Full text**")
            st.write(row["FullText"])

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Sentiment**")
                st.write(row["Sentiment"])
                if show_conf and isinstance(row["SentimentProbs"], (list, tuple)):
                    probs = row["SentimentProbs"]
                    bar = go.Figure(go.Bar(
                        x=probs, y=SENTIMENT_ORDER, orientation="h",
                        marker_color=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER]
                    ))
                    bar.update_layout(xaxis_range=[0,1], height=200, margin=dict(l=10,r=10,t=10,b=10))
                    st.plotly_chart(bar, use_container_width=True)

            with c2:
                st.markdown("**Severity**")
                st.write(row["Severity"])
                if show_conf and isinstance(row["SeverityProbs"], (list, tuple)):
                    probs2 = row["SeverityProbs"]
                    bar2 = go.Figure(go.Bar(
                        x=probs2, y=SEVERITY_ORDER, orientation="h",
                        marker_color=[SEVERITY_COLORS[s] for s in SEVERITY_ORDER]
                    ))
                    bar2.update_layout(xaxis_range=[0,1], height=200, margin=dict(l=10,r=10,t=10,b=10))
                    st.plotly_chart(bar2, use_container_width=True)

            st.markdown("**Top topics**")
            st.write(", ".join(row["TopTopicList"]) if row["TopTopicList"] else "—")

else:
    st.info("Upload a dataset to start analysis.")
