# LABR — Arabic Book Review Rating Prediction (1★–5★)

**TP2 — Transformers & LLMs · Master Big Data & IoT, ENSAM Casablanca**
**Groupe 3 — Bouaouda & El Amrani**

Predict the star rating (1–5) of an Arabic GoodReads review (LABR dataset).
The task is **ordinal**: predicting 4 for a true 5 is a smaller error than predicting 1.
Main metric: **macro-F1**. Secondary metric: **MAE** (mean star error).

## 1. Setup

### Local (conda)
```bash
conda env create -f environment.yml
conda activate labr-rating
python -m ipykernel install --user --name labr-rating --display-name "Python (labr-rating)"
```
Select the kernel **Python (labr-rating)** in Jupyter.

To update after a change in `requirements.txt`:
```bash
conda env update -f environment.yml --prune
```

### Google Colab (GPU phases C, D, E)
Colab already ships torch/transformers. Do **not** reinstall them; add only what is missing:
```python
!pip install -q arabert==1.0.1 gensim==4.3.3 statsmodels
```
Then upload `src/` and `data/` (or mount Google Drive).

## 2. Data

Download from the official LABR repository (`mohamedadaly/LABR`, folder `data/`):
- `reviews.tsv`
- `5class-unbalanced-train.txt`
- `5class-unbalanced-test.txt`

Place them in `data/`. The data is **not** committed to the repo.

## 3. Project structure
```
labr-rating/
├── data/            # raw LABR files + splits/ (generated)
├── src/             # reusable code: loading, normalization, metrics, models
├── notebooks/       # one notebook per phase (run in order)
├── results/         # metrics tables, figures (generated)
├── predictions/     # per-model predictions on val/test (generated)
├── tests/           # pytest unit tests
├── environment.yml
├── requirements.txt
└── README.md
```

## 4. Phases (run notebooks in order)

| Notebook | Phase | Content |
|---|---|---|
| `01_audit.ipynb` | 4.1 | Data audit, Arabic normalization, duplicates/leakage, train/val/test splits |
| `02_baselines.ipynb` | A | Majority class, length-only, TF-IDF + Logistic Regression |
| `03_zero_shot.ipynb` | B | Zero-shot NLI (mDeBERTa), no training |
| `04_neural_scratch.ipynb` | C | Embeddings + BiLSTM, attention, mini-Transformer |
| `05_bert_frozen.ipynb` | D | Frozen AraBERT features (CLS vs mean pooling) |
| `06_bert_finetune.ipynb` | E | Fine-tuned AraBERT (classification + regression) |
| `07_robustness.ipynb` | F | Unseen books/users, bootstrap CIs, McNemar test |
| `08_final.ipynb` | G | Final comparison: quality vs cost vs latency |

## 5. Reproducibility rules
- **Fixed seed = 42** for splits, sklearn, numpy and torch.
- **val** = 10% of the official train, stratified by rating. All choices are made on val.
- **test** is used **once**, in Phase F.
- Anything learned from data (vocabulary, TF-IDF, class weights) is fit on **train only**.
- Split IDs are saved in `data/splits/`, so every model uses exactly the same examples.

## 6. Tests
```bash
pytest tests/
```