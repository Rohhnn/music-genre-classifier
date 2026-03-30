import os
import numpy as np
import torch
import librosa
import streamlit as st
import plotly.graph_objects as go
from huggingface_hub import hf_hub_download
from transformers import AutoFeatureExtractor, ASTForAudioClassification

AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
HF_REPO_ID = "Rohhnn/music-genre-classifier"
HF_FILENAME = "ast_src_full_finetune_400_best.pth"

SR = 16000
DURATION = 10.0
N_CROPS = 5

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]

GENRE_EMOJI = {
    "blues":     "🎸",
    "classical": "🎻",
    "country":   "🤠",
    "disco":     "🕺",
    "hiphop":    "🎤",
    "jazz":      "🎷",
    "metal":     "🤘",
    "pop":       "🎵",
    "reggae":    "🌴",
    "rock":      "⚡",
}

DEVICE = torch.device("cpu")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    weights_path      = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    feature_extractor = AutoFeatureExtractor.from_pretrained(AST_MODEL_ID)
    model             = ASTForAudioClassification.from_pretrained(
        AST_MODEL_ID, num_labels=10, ignore_mismatched_sizes=True
    )
    for param in model.audio_spectrogram_transformer.parameters():
        param.requires_grad = False
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return feature_extractor, model


def predict_with_tta(model, feature_extractor, audio, sr, duration, n_crops, device):
    crop_len  = int(sr * duration)
    total_len = len(audio)
    probs_sum = torch.zeros(len(GENRES), device=device)

    if total_len <= crop_len:
        crops = [np.pad(audio, (0, crop_len - total_len))]
    else:
        starts = np.linspace(0, total_len - crop_len, n_crops, dtype=int)
        crops  = [audio[s: s + crop_len] for s in starts]

    for crop in crops:
        inputs       = feature_extractor(crop, sampling_rate=sr,
                                         return_tensors="pt", padding=True)
        input_values = inputs["input_values"].to(device)
        with torch.no_grad():
            logits = model(input_values).logits
        probs_sum += torch.softmax(logits, dim=1).squeeze(0)

    return (probs_sum / len(crops)).cpu().numpy()


st.set_page_config(page_title="Music Genre Classifier", page_icon="🎶", layout="centered")

st.title("🎶 Music Genre Classifier")
st.markdown(
    "Upload an audio file and the model will predict the most likely music genres. "
    "Predictions are averaged over 5 time-spaced crops for robustness."
)

feature_extractor, model = load_model()

uploaded_file = st.file_uploader("Upload a WAV or MP3 file", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    with st.spinner("Analysing audio..."):
        tmp_path = f"/tmp/{uploaded_file.name}"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.read())

        try:
            audio, _ = librosa.load(tmp_path, sr=SR, mono=True)
        except Exception as e:
            st.error(f"Could not load audio file: {e}")
            st.stop()

        probs    = predict_with_tta(model, feature_extractor, audio,
                                    SR, DURATION, N_CROPS, DEVICE)
        top5_idx = np.argsort(probs)[::-1][:5]

        top_genre = GENRES[top5_idx[0]]
        top_prob  = float(probs[top5_idx[0]])

        os.remove(tmp_path)

    st.markdown("---")
    st.markdown(
        f"### {GENRE_EMOJI[top_genre]} Predicted Genre: **{top_genre.capitalize()}**"
        f"\u2002\u2002`{top_prob * 100:.1f}% confidence`"
    )

    top5_genres = [GENRES[i] for i in top5_idx]
    top5_probs  = [float(probs[i]) for i in top5_idx]
    colors      = ["#1DB954"] + ["#4a4a6a"] * 4

    fig = go.Figure(go.Bar(
        x=[p * 100 for p in top5_probs[::-1]],
        y=[f"{GENRE_EMOJI[g]} {g.capitalize()}" for g in top5_genres[::-1]],
        orientation="h",
        marker_color=colors[::-1],
        text=[f"{p * 100:.1f}%" for p in top5_probs[::-1]],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="Probability (%)",
        xaxis=dict(range=[0, 115]),
        yaxis_title=None,
        margin=dict(l=20, r=40, t=20, b=20),
        height=280,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
with st.expander("Model details"):
    col1, col2, col3 = st.columns(3)
    col1.metric("Kaggle F1",         "0.9314")
    col2.metric("Val F1",            "0.9763")
    col3.metric("Samples per genre", "400")
    st.markdown(
        f"**Base model:** `{AST_MODEL_ID}`  \n"
        f"**Fine-tuning:** Full (all 87M parameters)  \n"
        f"**Epochs:** 10  \n"
        f"**Weights:** [`{HF_REPO_ID}`](https://huggingface.co/{HF_REPO_ID})"
    )

st.markdown("---")
st.markdown(
    "<small>Built for the IIT Madras BS Degree DL & GenAI Project — "
    "[Messy Mashup Kaggle Competition]"
    "(https://www.kaggle.com/competitions/jan-2026-dl-gen-ai-project)</small>",
    unsafe_allow_html=True,
)