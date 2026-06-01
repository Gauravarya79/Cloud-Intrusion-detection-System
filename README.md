# Cloud-Based Intrusion Detection System

A machine learning-powered Network Intrusion Detection System (NIDS) that classifies 
network traffic as Normal, DDoS, Brute Force, or Restricted Access using a hybrid 
ML pipeline built on the CICIDS 2017 dataset.

---

## Tech Stack

Python, scikit-learn, Streamlit, Plotly, CICIDS 2017 Dataset

---

## ML Pipeline

Raw CICIDS Dataset → Preprocessing → Chi-Squared + RFE Feature Selection → Random Forest Classifier → Streamlit Dashboard

---

## Features

- Hybrid feature selection using Chi-Squared Filter and RFE
- Random Forest classifier for multi-class traffic detection
- Stratified sampling to handle class imbalance
- Interactive Streamlit dashboard with real-time Plotly visualizations
- Label leakage-free pipeline for unbiased model evaluation

---

## Setup

```bash
git clone https://github.com/Gauravarya79/Cloud-Based-Intrusion-Detection-System.git
cd Cloud-Based-Intrusion-Detection-System
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

Download CICIDS 2017 from: https://www.unb.ca/cic/datasets/ids-2017.html  
Place CSV files in the dataset/ folder before running.

---

## Authors

Gaurav Arya, Harshit, Dhruv Joshi, Shaurya Pandey, Samarth Singh, Ankit Vishnoi

Graphic Era (Deemed to be University) | B.Tech CSE Cyber Security | 2022–2026
