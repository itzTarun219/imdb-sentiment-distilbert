# IMDb Sentiment Classification with Fine-Tuned DistilBERT

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/itzTarun219/imdb-sentiment-distilbert/blob/main/Salla_Tarun_Kumar_23082746_LLM_Assignment3.ipynb)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c)
![Transformers](https://img.shields.io/badge/Transformers-5.14-yellow)

A reproducible comparison of TF-IDF baselines and a fine-tuned DistilBERT model for binary sentiment classification on the Stanford IMDb movie-review dataset.

## Project details

- **Student:** Salla Tarun Kumar
- **Student number:** 23082746
- **Module:** Research Methods in Data Science (7PAM2015)
- **Dataset:** Stanford IMDb Movie Reviews via Hugging Face
- **Transformer model:** `distilbert/distilbert-base-uncased`
- **Primary selection metric:** F1-score

### Research question

> How effectively does a fine-tuned DistilBERT model classify IMDb movie reviews as positive or negative compared with tuned traditional text-classification baselines?

## Experimental design

The study compares:

1. TF-IDF + Logistic Regression
2. TF-IDF + Multinomial Naïve Bayes
3. Fine-tuned DistilBERT

The workflow includes duplicate and cross-split leakage removal, balanced stratified subsets, pipeline-based baseline tuning, validation-only transformer tuning, a custom PyTorch training loop, bootstrap confidence intervals, McNemar's exact test, qualitative error analysis, and input-truncation analysis.

Final experimental splits:

| Split | Reviews | Negative | Positive |
|---|---:|---:|---:|
| Training | 12,000 | 6,000 | 6,000 |
| Validation | 3,000 | 1,500 | 1,500 |
| Test | 5,000 | 2,500 | 2,500 |

## Final test results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Fine-tuned DistilBERT | **0.9064** | **0.9093** | **0.9028** | **0.9061** | **0.9689** |
| TF-IDF + Logistic Regression | 0.8926 | 0.8915 | 0.8940 | 0.8928 | 0.9583 |
| TF-IDF + Multinomial Naïve Bayes | 0.8714 | 0.8883 | 0.8496 | 0.8685 | 0.9424 |

DistilBERT produced the highest value for every reported test metric. McNemar's exact test comparing DistilBERT with Logistic Regression returned `p = 0.004119`. Approximately 39.82% of test reviews exceeded the 256-token limit; F1 was 0.9231 for reviews within the limit and 0.8800 for longer reviews.

## Model architecture

![DistilBERT architecture](figures/distilbert_architecture.png)

## Repository structure

```text
.
├── Salla_Tarun_Kumar_23082746_LLM_Assignment3.ipynb
├── README.md
├── requirements.txt
├── .gitignore
├── figures/
│   ├── distilbert_architecture.png
│   ├── model_metric_comparison.png
│   ├── roc_curve_comparison.png
│   ├── training_validation_f1.png
│   └── training_validation_loss.png
└── results/
    ├── final_model_comparison.csv
    ├── bootstrap_confidence_intervals.csv
    └── truncation_analysis.csv
```

## Reproduce the analysis

The notebook is designed for Google Colab with a GPU runtime.

1. Open the notebook using the Colab badge above.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run the notebook from top to bottom.
4. Generated outputs are written to `/content/Salla_Tarun_Kumar_23082746_LLM_outputs/`.

The Hugging Face dataset and pretrained checkpoint are downloaded automatically. The complete run requires substantial GPU time because it includes baseline grid searches, transformer tuning, three-epoch fine-tuning, bootstrap resampling, and final evaluation.

## Key implementation features

- deterministic seed configuration;
- minimal sentiment-preserving text cleaning;
- explicit data-leakage assertions;
- TF-IDF pipelines evaluated with stratified cross-validation;
- custom PyTorch `Dataset`, `DataLoader`, training and evaluation loops;
- AdamW optimisation, linear warm-up/decay, mixed precision and gradient clipping;
- validation-F1 checkpoint selection;
- accuracy, precision, recall, F1 and ROC-AUC evaluation;
- confusion matrices, ROC curves and learning curves;
- bootstrap confidence intervals and McNemar's paired test;
- high-confidence error inspection and truncation analysis.

## Limitations

The experiment uses one domain, one random seed, bounded hyperparameter tuning, balanced subsets rather than the full labelled dataset, and a 256-token input limit. Transformer results may also vary slightly across GPU environments because some accelerated attention operations are not fully deterministic.

## Academic-use notice

This repository accompanies an individual University of Hertfordshire assessment. It is published for review and reproducibility. Any reuse must comply with applicable academic-integrity requirements and properly acknowledge the original sources.
