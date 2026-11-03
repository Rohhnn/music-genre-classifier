# Music Genre Classifier — Streamlit app

Thin Streamlit front-end for the AST music genre classifier. Model weights load at runtime from Hugging Face (`Rohhnn/music-genre-classifier`).

This folder is a **mirror** of the Cloud deploy repo:

- **Live demo:** [Streamlit app](https://music-genre-classifier-app-nxjyfsazbfkpqqzwpujfyw.streamlit.app/)
- **Deploy / live app source:** [Rohhnn/music-genre-classifier-app](https://github.com/Rohhnn/music-genre-classifier-app)
- **Full project (training, notebooks, report):** [Rohhnn/music-genre-classifier](https://github.com/Rohhnn/music-genre-classifier) *(after rename)*

## Run locally

```bash
pip install -r requirements.txt streamlit huggingface_hub
streamlit run app.py
```

## Model

| | |
|---|---|
| Base | `MIT/ast-finetuned-audioset-10-10-0.4593` |
| Fine-tune | Full AST, 400 samples/genre |
| Kaggle macro F1 | 0.9314 |
| Weights | [huggingface.co/Rohhnn/music-genre-classifier](https://huggingface.co/Rohhnn/music-genre-classifier) |
