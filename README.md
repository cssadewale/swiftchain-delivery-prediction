# 🚚 SwiftChain Delivery Delay Prediction

> **Live App →** [your-app-url.streamlit.app](https://your-app-url.streamlit.app)
> *(Replace this link with your actual Streamlit URL after deployment)*

---

## Project Summary

A multiclass classification model that predicts — at the moment of dispatch — whether a logistics order will arrive **Late**, **On-Time**, or **Early**, based on order and shipping characteristics.

Built for **SwiftChain Analytics**, a global logistics intelligence firm operating across five markets: Europe, LATAM, USCA, Pacific Asia, and Africa.

---

## The Business Problem

Late deliveries trigger customer service contacts, refund requests, and churn. Early deliveries represent over-investment in speed. Knowing which outcome an order is headed for enables three operational levers:
1. **Proactive customer communication** — notify customers before they complain
2. **Targeted intervention** — escalate high-risk shipments before they miss the window
3. **Smarter shipping mode selection** — route decisions driven by data, not habit

---

## Key Results

| Metric | Value |
|--------|-------|
| Best Model | Gradient Boosting Classifier (Tuned) |
| Test Accuracy | 62.0% |
| Weighted F1 | 0.5791 |
| CV F1 (5-fold) | 0.5768 ± 0.0091 |
| Late Recall | 68.1% |
| Early Recall | 78.0% |
| Top Feature | `shipping_mode_Standard Class` (67.2% importance) |

---

## Key Finding

**Standard Class shipping mode is the single dominant risk vector.** It accounts for **67.2% of total model importance** and carries a **38.1% late-delivery rate** — nearly 2× the global average of 22.8%.

Shifting just 10% of Standard Class orders (~910 per period) to Second Class (which has only a 0.3% late rate) would prevent approximately **347 late deliveries per period** — a direct, quantified operational impact.

---

## Model Hyperparameters

Selected by GridSearchCV (12 combinations × 3-fold CV = 36 fits):

| Parameter | Value |
|-----------|-------|
| `learning_rate` | 0.01 |
| `max_depth` | 3 |
| `n_estimators` | 200 |

---

## Feature Importance (Top 5)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `shipping_mode_Standard Class` | 67.15% |
| 2 | `shipping_duration` | 17.94% |
| 3 | `shipping_mode_Second Class` | 6.34% |
| 4 | `shipping_mode_Same Day` | 6.06% |
| 5+ | All remaining 305 features | < 0.42% each |

Top 2 features alone account for **85.09%** of all model learning.

---

## Tech Stack

| Area | Tools |
|------|-------|
| Language | Python 3.11 |
| Data manipulation | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine learning | Scikit-learn |
| Model serialisation | Joblib |
| Web deployment | Streamlit |

---

## Repository Structure

```
swiftchain-delivery-prediction/
│
├── SwiftChain_Delivery_Delay_Prediction_Portfolio.ipynb   ← Full analysis notebook
├── app.py                                                  ← Streamlit web application
├── swiftchain_delay_predictor.pkl                          ← Trained model
├── swiftchain_scaler.pkl                                   ← Fitted StandardScaler
├── requirements.txt                                        ← Python dependencies
├── .gitignore                                              ← Files to exclude from git
└── README.md                                               ← This file
```

---

## Project Workflow

| Step | Phase |
|------|-------|
| 1 | Data Loading & Inspection — 15,549 orders × 41 features |
| 2 | Data Cleaning — datetime conversion, categorical consistency |
| 3 | Exploratory Data Analysis — Uni/Bivariate/Multivariate |
| 4 | Preprocessing — IQR capping, One-Hot Encoding |
| 5 | Feature Engineering — `shipping_duration` derived feature |
| 6 | Model Training — Logistic Regression, Random Forest, Gradient Boosting |
| 7 | Hyperparameter Tuning — GridSearchCV (36 fits) |
| 8 | Feature Importance & Serialisation |
| 9 | Streamlit Deployment |

---

## Author

**Adewale Samson Adeagbo**  
Lead Data Scientist / ML Engineer · Lagos, Nigeria

| | |
|---|---|
| 📧 Email | buildingmyictcareer@gmail.com |
| 📱 Phone | +2348100866322 |
| 💼 LinkedIn | [linkedin.com/in/adewalesamsonadeagbo](https://linkedin.com/in/adewalesamsonadeagbo) |
| 🐙 GitHub | [github.com/cssadewale](https://github.com/cssadewale) |
| 🌐 Website | [hmgconcepts.business.site](https://hmgconcepts.business.site) |

---

*Part of a supervised ML portfolio — binary classification, multiclass classification, and regression projects. Each includes a live Streamlit deployment.*
