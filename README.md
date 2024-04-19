# FinTech Industrial Project
Include some key components for industrial project

## Product Related Comment Scraping
Notebook used to scrape related product comments on App Store/Google Play and public investment forum. [For research purpose only]

## Sentiment Analysis
Fine-tuned a pre-train model with scaped dataset in cantonese with self-supervised learning. \
For efficient training and avoid catastrophic forgetting:
- self-supervised learning on last classification layer only
- introduce weighted loss function for class imbalance
- full parameters update on a small set of mannually annotated samples

## Web Application
Streamlit application for communication and demostration
