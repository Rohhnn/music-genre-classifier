# DL & GenAI Project — Messy Mashup

**Name:** Rohan Puntambekar  
**Roll No.:** 21f3002298  
**Course:** Intro to DL and GenAI — IIT Madras BS Degree (Jan 2026)  
**Competition:** [Messy Mashup on Kaggle](https://www.kaggle.com/competitions/jan-2026-dl-gen-ai-project)  

---

## Task

10-class music genre classification (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock) from stem-based audio mashups. The training set consists of 1,000 songs (100 per genre), each decomposed into four stems: drums, vocals, bass, and other. The test set contains 3,020 cross-song mashups with added ESC-50 noise. Evaluation metric: macro F1 score.

---

## Repository Structure

```
dl-genai-project-25-t3/
├── .gitignore
├── README.md
├── requirements.txt
├── notebooks/                         # Course milestone notebooks
│   ├── milestone_1_notebook.ipynb
│   ├── milestone2_notebook_eda.ipynb
│   ├── milestone2_notebook_decision_trees.ipynb
│   ├── milestone3_notebook_CNN.ipynb
│   ├── milestone4_crnn_training.ipynb
│   ├── milestone5_ast_training.ipynb
│   ├── helper_functions.py
│   └── model_skeleton.py
├── scripts/                         # EDA + training notebooks (main pipeline)
│   ├── eda_and_preprocess.ipynb
│   ├── train_crnn.ipynb
│   ├── train_ast.ipynb
│   ├── train_efficientB.ipynb
│   ├── train_mert_colab.ipynb
│   ├── train_ast_full_finetune_colab.ipynb
│   └── wandb/                       # local Weights & Biases runs (gitignored)
├── inference/                       # Kaggle / submission inference notebook
│   └── dl-21f3002298-notebook-t12026.ipynb
├── deployment/                      # Streamlit demo app
│   ├── app.py
│   └── requirements.txt
├── reports/                         # Final report materials
├── models/                          # Saved checkpoints & figures (largely gitignored)
├── working/                         # Feature caches & intermediate artifacts (gitignored)
├── data/                            # Competition dataset (gitignored)
│   └── messy_mashup/
│       ├── genres_stems/
│       ├── mashups/
│       ├── ESC-50-master/
│       ├── test.csv
│       └── sample_submission.csv
└── venv/                            # Local Python environment (gitignored)
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Rohhnn/dl-genai-project-25-t3.git
cd dl-genai-project-25-t3
```

### 2. Create and activate the conda environment

```bash
conda create -n gpu_env python=3.9
conda activate gpu_env
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU support, install PyTorch separately with the appropriate CUDA version for your system. See [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the correct command. The project was developed with CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Data

Place the competition dataset under `data/messy_mashup/` with the following structure:

```
data/messy_mashup/
├── genres_stems/       # 10 genres × 100 songs × 4 stems
├── mashups/            # 3020 test mashup WAV files
├── ESC-50-master/      # ESC-50 noise dataset
├── test.csv
└── sample_submission.csv
```

---

## Running Training

Training notebooks live under `scripts/`. Open them in Jupyter or any suitable IDE and run cells sequentially.

For MERT and full AST fine-tuning, use `scripts/train_mert_colab.ipynb` and `scripts/train_ast_full_finetune_colab.ipynb` on Google Colab (T4 GPU recommended).

---

## Deployment

The best model (AST full fine-tune, 400 samples/genre) is deployed as a Streamlit app. To run locally:

```bash
cd deployment
pip install -r requirements.txt
streamlit run app.py
```