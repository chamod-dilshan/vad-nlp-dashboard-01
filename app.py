import streamlit as st
import pandas as pd
import plotly.express as px
from utils.loader import load_all_models
from utils.inference import predict_single_label, predict_multilabel_topk

st.set_page_config(page_title="VAD NLP Insights", layout="wide")

st.title("🫀 Extracting Safety Insights from VAD Reports (NLP Dashboard)")

# --- Load models
(models_sent, models_sev, models_top) = load_all_models()

tok_s, mdl_s, meta_s = models_sent
tok_v, mdl_v, meta_v = models_sev
tok_t, mdl_t, meta_t = models_top

st.sidebar.header("Upload Excel")
uploaded_file = st.sidebar.file_uploader("Upload a file with FOI_TEXT column", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {len(df)} rows.")

    if "FOI_TEXT" not in df.columns:
        st.error("Missing column FOI_TEXT.")
    else:
        texts = df["FOI_TEXT"].fillna("").astype(str).tolist()

        results = []
        for text in texts:
            s_pred, s_probs = predict_single_label(tok_s, mdl_s, text)
            sev_pred, sev_probs = predict_single_label(tok_v, mdl_v, text)
            top_preds, top_scores = predict_multilabel_topk(
                tok_t, mdl_t, text, meta_t["label_space"], meta_t["topic_labels"]
            )
            results.append({
                "Text": text[:200] + ("..." if len(text)>200 else ""),
                "Sentiment": meta_s["id2label"][str(s_pred)],
                "Severity": meta_v["id2label"][str(sev_pred)],
                "Top Topics": ", ".join([t[1] for t in top_preds])
            })

        res_df = pd.DataFrame(results)
        st.dataframe(res_df, use_container_width=True)

        # Visualization
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sentiment Distribution")
            fig = px.histogram(res_df, x="Sentiment", color="Sentiment")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Severity Distribution")
            fig2 = px.histogram(res_df, x="Severity", color="Severity")
            st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Upload a dataset to start analysis.")
