# Music Genre Classifier

10-class music genre classification from stem-based audio mashups

**Author:** Rohan Puntambekar

| | |
|---|---|
| **Live demo** | [Streamlit app](https://music-genre-classifier-app-nxjyfsazbfkpqqzwpujfyw.streamlit.app/) |
| **Deployment repo** | [`music-genre-classifier-app`](https://github.com/Rohhnn/music-genre-classifier-app) |
| **Model weights** | [`Rohhnn/music-genre-classifier`](https://huggingface.co/Rohhnn/music-genre-classifier) on Hugging Face |

> The Streamlit Cloud deployment is kept in a separate public repository so the live app stays stable while this project repo evolves. Local `deployment/` here mirrors that app.

## Overview

Training data: 1,000 songs (100 per genre), each split into four stems (drums, vocals, bass, other). Test set: 3,020 cross-song mashups with ESC-50 noise. Metric: **macro F1**.

Genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock.

### Results (best model)

| Model | Setup | Kaggle macro F1 | Val F1 |
|-------|--------|-----------------|--------|
| **AST (full fine-tune)** | 400 samples/genre, 10 epochs, TTA | **0.9314** | 0.9763 |

Other architectures explored: Decision Tree baseline, CNN, CRNN, EfficientNet, MERT, and staged AST fine-tuning. See `scripts/` and `reports/final_report.pdf`.

## Repository structure

```
music-genre-classifier/
├── notebooks/          # Course milestone notebooks (EDA → AST)
├── scripts/            # Main EDA + training pipeline
├── inference/          # Kaggle submission notebook
├── deployment/         # Streamlit app (mirror of deploy repo)
├── reports/            # Final report
├── models/             # Figures + local checkpoints (*.pth gitignored)
├── working/            # Feature caches (gitignored)
├── data/               # Competition dataset (gitignored)
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/Rohhnn/music-genre-classifier.git
cd music-genre-classifier
```

```bash
conda create -n gpu_env python=3.9
conda activate gpu_env
pip install -r requirements.txt
```

For GPU PyTorch (project used CUDA 11.8):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

See [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for other platforms.

### Data

Place the competition dataset under `data/messy_mashup/`:

```
data/messy_mashup/
├── genres_stems/       # 10 genres × 100 songs × 4 stems
├── mashups/            # 3020 test mashup WAV files
├── ESC-50-master/      # ESC-50 noise dataset
├── test.csv
└── sample_submission.csv
```

## Training

Open notebooks under `scripts/` and run cells in order.

Colab (T4 recommended): `scripts/train_mert_colab.ipynb`, `scripts/train_ast_full_finetune_colab.ipynb`.

Checkpoints are written under `models/` (not committed). Production weights live on Hugging Face.

## Deployment

### Live app

Hosted from [`Rohhnn/music-genre-classifier-app`](https://github.com/Rohhnn/music-genre-classifier-app). The app downloads `ast_src_full_finetune_400_best.pth` from Hugging Face at runtime.

### Run locally

```bash
cd deployment
pip install -r requirements.txt streamlit huggingface_hub
streamlit run app.py
```

## License / data

Competition audio and ESC-50 are **not** redistributed here. Obtain them from the competition / original sources under their terms.

## Academic context

Developed for *Intro to DL and GenAI* (IIT Madras BS Degree, Jan 2026 term) as the Messy Mashup project. Milestone notebooks under `notebooks/` reflect the course sequence; `scripts/` is the consolidated training path.
