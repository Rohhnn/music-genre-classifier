import os
import numpy as np
import torch
import librosa
import gradio as gr
from huggingface_hub import hf_hub_download
from transformers import AutoFeatureExtractor, ASTForAudioClassification

AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
HF_REPO_ID   = "Rohhnn/music-genre-classifier"
HF_FILENAME  = "ast_src_full_finetune_400_best.pth"

SR       = 16000
DURATION = 10.0
N_CROPS  = 5

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


feature_extractor, model = load_model()


def predict_with_tta(audio_array, sr):
    crop_len  = int(sr * DURATION)
    total_len = len(audio_array)
    probs_sum = torch.zeros(len(GENRES), device=DEVICE)

    if total_len <= crop_len:
        crops = [np.pad(audio_array, (0, crop_len - total_len))]
    else:
        starts = np.linspace(0, total_len - crop_len, N_CROPS, dtype=int)
        crops  = [audio_array[s: s + crop_len] for s in starts]

    for crop in crops:
        inputs       = feature_extractor(crop, sampling_rate=sr,
                                         return_tensors="pt", padding=True)
        input_values = inputs["input_values"].to(DEVICE)
        with torch.no_grad():
            logits = model(input_values).logits
        probs_sum += torch.softmax(logits, dim=1).squeeze(0)

    return (probs_sum / len(crops)).cpu().numpy()


def classify(audio_path):
    if audio_path is None:
        return {}, "Please upload an audio file."

    try:
        audio, _ = librosa.load(audio_path, sr=SR, mono=True)
    except Exception as e:
        return {}, f"Could not load audio: {e}"

    probs    = predict_with_tta(audio, SR)
    top5_idx = np.argsort(probs)[::-1][:5]

    label_probs = {
        f"{GENRE_EMOJI[GENRES[i]]} {GENRES[i].capitalize()}": float(probs[i])
        for i in top5_idx
    }
    top_genre   = GENRES[top5_idx[0]]
    top_prob    = probs[top5_idx[0]]
    summary     = f"{GENRE_EMOJI[top_genre]} **{top_genre.capitalize()}** — {top_prob*100:.1f}% confidence"
    return label_probs, summary


MODEL_INFO = f"""
### Model details

| | |
|---|---|
| **Kaggle F1** | 0.9314 |
| **Val F1** | 0.9763 |
| **Training samples** | 400 per genre |
| **Base model** | `{AST_MODEL_ID}` |
| **Fine-tuning** | Full (all 87M parameters) |
| **Epochs** | 10 |
| **Weights** | [Rohhnn/music-genre-classifier](https://huggingface.co/Rohhnn/music-genre-classifier) |

---

Built for the IIT Madras BS Degree DL & GenAI Project —
[Messy Mashup Kaggle Competition](https://www.kaggle.com/competitions/jan-2026-dl-gen-ai-project)
"""

with gr.Blocks(title="Music Genre Classifier") as demo:
    gr.Markdown("# 🎶 Music Genre Classifier")
    gr.Markdown(
        "Upload a WAV or MP3 file. The model predicts the top 5 most likely genres "
        "by averaging softmax probabilities across 5 evenly spaced 10-second crops."
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Upload audio file")
            run_btn     = gr.Button("Classify", variant="primary")
        with gr.Column():
            summary_out = gr.Markdown(label="Top prediction")
            chart_out   = gr.Label(num_top_classes=5, label="Top 5 genres")

    run_btn.click(fn=classify, inputs=audio_input, outputs=[chart_out, summary_out])

    gr.Markdown(MODEL_INFO)

demo.launch()