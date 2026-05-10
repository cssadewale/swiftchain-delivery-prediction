# 🚚 SwiftChain Delivery Delay Prediction

<div align="center">

[![Live App](https://img.shields.io/badge/🚀%20Live%20App-Streamlit-00c896?style=for-the-badge)](https://adewale-swiftchain-delivery-prediction.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-cssadewale-181717?style=for-the-badge&logo=github)](https://github.com/cssadewale)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Adewale%20Adeagbo-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/adewalesamsonadeagbo)
[![3MTT](https://img.shields.io/badge/3MTT-Cohort%203%20Fellow-006400?style=for-the-badge)](https://3mtt.nitda.gov.ng/)

**🔗 Live App → [adewale-swiftchain-delivery-prediction.streamlit.app](https://adewale-swiftchain-delivery-prediction.streamlit.app/)**

</div>

---

## 📌 Programme Context

This project is the **4th and final capstone project** of the **Data Science & Machine Learning track** delivered on the **[Darey.io](https://darey.io)** training platform, under the **3 Million Technical Talent (3MTT) Programme** — a flagship initiative of the **Federal Ministry of Communications, Innovation & Digital Economy**, Government of the Federal Republic of Nigeria.

### About 3MTT

The **3 Million Technical Talent (3MTT) Programme** is Nigeria's largest government-funded digital skills initiative, launched in December 2023 as a central pillar of President Bola Ahmed Tinubu's **Renewed Hope Agenda**. Executed in collaboration with the **National Information Technology Development Agency (NITDA)**, the programme targets training **3 million Nigerians** in high-demand technical skills — including Data Science & AI/ML, Software Development, Cloud Computing, Cybersecurity, UI/UX Design, and Product Management — by the year 2027.

Key facts about 3MTT:
- **1.8 million+ applications** received across all cohorts
- **90,000+ learners** onboarded through Cohorts 1 and 2; **35,000+ currently enrolled** in Cohort 3
- **7,500+ fellows** have secured employment through the programme's employer networks
- **2,000+ real-world projects** completed by fellows addressing community and industry problems
- Backed by partners including **MTN Nigeria (₦3 billion committed)**, **AWS (Education Equity Initiative)**, and **120+ Applied Learning Cluster organisations** across all 36 states and the FCT
- Described by the World Economic Forum as **"the largest known talent accelerator in the world"**

### About Darey.io

**[Darey.io](https://darey.io)** is a Nigerian ed-tech company and official **3MTT training provider**, founded in 2021 by Dare Olufunmilayo and headquartered in Yaba, Lagos. Darey.io was selected by the Federal Government as one of its lead training partners to implement the 3MTT programme at scale.

The platform distinguishes itself through:
- **Project-based learning** — every skill is taught through real, deliverable projects rather than passive video consumption
- **Proprietary Career Score Algorithm** — tracks engagement, project quality, live class attendance, quiz performance, and peer contribution; fellows must achieve a score of **600+** for certification and **800+** for job placement eligibility
- **Dual-platform model** — Darey.io handles skill acquisition; the companion platform **Xterns** provides industry-grade practical experience and portfolio building
- **Curriculum depth** — courses in DevOps, Cloud Computing, Data Science, Cybersecurity, Product Management, and AI for various industries
- **Fully funded for 3MTT fellows** — the Federal Government covers 100% of training costs for enrolled beneficiaries

### My Participation

I am **Adewale Samson Adeagbo**, a **Cohort 3 fellow** of the 3MTT DSML (Data Science & Machine Learning) Track, delivered via the Darey.io platform. The four capstone projects completed across this programme are:

| # | Project | Type | Live App |
|---|---------|------|----------|
| 1 | Yakub Trading Group — Staff Promotion Prediction | Binary Classification | [View App](https://yakub-promotion-prediction.streamlit.app) |
| 2 | Insurance Claim Prediction | Regression | [View App](https://adewale-insurance-claim-prediction.streamlit.app) |
| 3 | Bank Customer Churn Prediction | Binary Classification | [View App](https://adewale-bank-customer-churn-prediction.streamlit.app) |
| **4** | **SwiftChain Delivery Delay Prediction** | **Multiclass Classification** | **[View App ↗](https://adewale-swiftchain-delivery-prediction.streamlit.app/)** |

---

## 🎯 Project Overview

**SwiftChain Analytics** is a global logistics intelligence firm that partners with e-commerce platforms and fulfilment centres across five markets: Europe, LATAM, USCA, Pacific Asia, and Africa.

The core business challenge: predict — **at the moment of dispatch** — whether a given order will arrive **Late**, **On-Time**, or **Early**. Late deliveries trigger customer service contacts, refund requests, and churn. Early deliveries, while positive, represent over-investment in speed. Accurate prediction enables three operational levers:

1. **Proactive customer communication** — notify customers before they chase
2. **Targeted escalation** — intervene on high-risk shipments before the window closes
3. **Smarter mode selection** — route decisions driven by data, not habit

---

## ❓ Business Problem

> *Can we accurately predict — at the moment of dispatch — whether an order will arrive Late, On-Time, or Early, and identify which operational factors most directly control delivery outcomes?*

**Target Variable — `label` (3 classes):**

| Label | Meaning | Count in Dataset | Share |
|-------|---------|-----------------|-------|
| **−1** | Order arrived **Late** | 3,544 | 22.8% |
| **0** | Order arrived **On-Time** | 3,028 | 19.5% |
| **1** | Order arrived **Early** | 8,977 | 57.7% |

Class imbalance (Early is 2.5× more frequent than On-Time) is a key modelling challenge addressed in preprocessing and evaluation.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | Darey.io / 3MTT DSML Programme — Project 4 |
| Total Records | 15,549 orders |
| Raw Features | 41 columns |
| Date Range | 2015 – 2018 |
| Markets | Europe, LATAM, USCA, Pacific Asia, Africa |
| Customer Segments | Consumer, Corporate, Home Office |
| Shipping Modes | Standard Class, Second Class, First Class, Same Day |

**Key feature categories:**
- **Order logistics:** shipping mode, market, order date, shipping date
- **Customer profile:** segment, location city/state/country
- **Product details:** category, product name, quantity, price
- **Financials:** sales, discount, profit, order item profit ratio
- **Engineered:** `shipping_duration` (days between order placement and dispatch)

---

## 🔬 Project Workflow

| Step | Phase | Key Actions |
|------|-------|-------------|
| **1** | Data Loading & Inspection | Loaded 15,549 records; inspected shape, dtypes, missing values, target distribution |
| **2** | Data Cleaning | Standardised column names to snake_case; corrected datetime types; resolved missing values; removed duplicates; validated categorical consistency |
| **3** | Exploratory Data Analysis | Univariate (distributions, skewness); bivariate (correlations, target relationships); multivariate (shipping mode × market × label) |
| **4** | Data Preprocessing | IQR outlier capping; one-hot encoding of 9 categorical features; StandardScaler on numerical features |
| **5** | Feature Engineering | Derived `shipping_duration` from `order_date` and `shipping_date`; preliminary Random Forest importance ranking to shortlist predictors |
| **6** | Model Development | Trained and compared: Logistic Regression, Random Forest, Gradient Boosting |
| **7** | Hyperparameter Tuning | GridSearchCV across 12 combinations × 3-fold CV = 36 total fits on Gradient Boosting |
| **8** | Evaluation | Accuracy, Weighted F1, per-class Precision/Recall/F1, confusion matrix, 5-fold cross-validation |
| **9** | Feature Importance | Gini impurity importance from tuned Gradient Boosting; top 2 features account for 85% of model learning |
| **10** | Serialisation | Model saved as `swiftchain_delay_predictor.pkl`; scaler saved as `swiftchain_scaler.pkl` via joblib |
| **11** | Deployment | Streamlit web app deployed to Streamlit Community Cloud |

---

## ⚙️ Data Preprocessing Pipeline

```
Raw Data (15,549 × 41)
        │
        ├── Column name standardisation (snake_case)
        ├── Datetime conversion (order_date, shipping_date → datetime64)
        ├── Missing value treatment (median / mode imputation per column)
        ├── Duplicate removal
        │
        ├── Feature Engineering
        │       └── shipping_duration = shipping_date − order_date (days)
        │
        ├── Outlier Treatment — IQR Capping
        │       └── Target variable (label) excluded from capping
        │
        ├── Categorical Encoding (OneHotEncoder, drop='first')
        │       └── 9 categorical columns → expanded feature matrix
        │
        ├── Feature Scaling (StandardScaler)
        │       ├── fit_transform on X_train only
        │       └── transform on X_test  ← no data leakage
        │
        └── Train / Test Split (80% / 20%, random_state=42)
                ├── X_train: 12,439 samples × 309 features
                └── X_test:   3,110 samples × 309 features
```

> **Note on 309 features:** One-hot encoding of 9 categorical columns — some with many unique values such as city names and product names — expands the raw feature matrix to 309 columns. This explains why the top 2 features dominate; most of the 307 remaining OHE columns carry negligible discriminating signal.

---

## 🤖 Model Development & Comparison

Three models were trained on the full preprocessed feature matrix and evaluated on the held-out test set:

| Model | Accuracy | Weighted F1 | Notes |
|-------|:--------:|:-----------:|-------|
| Logistic Regression | 0.5682 | 0.5114 | Weakest — confirms non-linear class boundaries |
| Random Forest | 0.5839 | 0.5214 | Stronger — ensemble handles non-linearity |
| **Gradient Boosting** | **0.6174** | **0.5794** | ✅ Best — selected for tuning |

**Why Gradient Boosting won:** Its sequential, error-correcting architecture is well-suited to the asymmetric class structure of this dataset. Each tree corrects the residual errors of the previous stage, making it more effective at capturing the non-linear boundary between Late, On-Time, and Early orders than both parallel ensembles (Random Forest) and linear classifiers (Logistic Regression).

---

## 🔧 Hyperparameter Tuning — GridSearchCV

```python
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth':     [3, 5],
    'n_estimators':  [100, 200]
}
# 3 × 2 × 2 = 12 combinations × 3-fold CV = 36 total fits
```

**Best parameters found:**

| Parameter | Value | Interpretation |
|-----------|:-----:|---------------|
| `learning_rate` | **0.01** | Small step size — reduces overfitting to noise |
| `max_depth` | **3** | Shallow trees — prevents memorising low-signal features |
| `n_estimators` | **200** | More stages compensate for the conservative learning rate |

The tuner selected the most regularised configuration. When a single binary feature dominates 67% of model importance, deep trees (depth 5+) overfit to the remaining low-signal features. Shallow trees with a low learning rate and more stages is the theoretically correct trade-off in this scenario.

---

## 📈 Final Model Performance

### Overall Metrics

| Metric | Baseline GB | Tuned GB | Change |
|--------|:-----------:|:--------:|:------:|
| Accuracy | 0.6174 | **0.6199** | +0.25 pp ✅ |
| Weighted F1 | 0.5794 | **0.5791** | −0.03 pp |

### Per-Class Performance (Tuned Model)

| Class | Precision | Recall | F1-Score | Support |
|-------|:---------:|:------:|:--------:|:-------:|
| **Late (−1)** | 0.45 | **0.68** | 0.54 | 709 |
| **On-Time (0)** | 0.68 | 0.07 | 0.13 | 606 |
| **Early (1)** | **0.71** | **0.78** | **0.74** | 1,795 |
| **Weighted Avg** | **0.65** | **0.62** | **0.58** | **3,110** |

### Confusion Matrix

```
                   Predicted
                  Late  On-Time  Early
Actual  Late       483      —      226
        On-Time    192     45      369
        Early       63    332    1,400
```

**Interpretation:**
- **Late Recall = 68.1%** — the model flags 2 in every 3 genuinely late orders. For proactive customer outreach, this is operationally actionable even at a precision of 0.45 (45 true positives per 100 flagged orders).
- **On-Time Recall = 7.4%** — the primary model weakness. On-Time orders sit at the feature boundary between Late and Early, making them the hardest class to isolate.
- **Early Recall = 78.0%** — the model's strongest class, consistent with Early being the dataset majority (57.7% of all orders).

### 5-Fold Cross-Validation

| Fold | Weighted F1 |
|------|:-----------:|
| Fold 1 | 0.5710 |
| Fold 2 | 0.5892 |
| Fold 3 | 0.5796 |
| Fold 4 | 0.5813 |
| Fold 5 | 0.5628 |
| **Mean ± Std** | **0.5768 ± 0.0091** |

The narrow standard deviation (0.0091) confirms consistent generalisation across different data subsets. The test-set Weighted F1 of 0.5791 falls comfortably within the CV confidence band.

---

## 🔑 Feature Importance

| Rank | Feature | Importance | Cumulative |
|------|---------|:----------:|:----------:|
| 1 | `shipping_mode_Standard Class` | **67.15%** | 67.15% |
| 2 | `shipping_duration` | **17.94%** | **85.09%** |
| 3 | `shipping_mode_Second Class` | 6.34% | 91.43% |
| 4 | `shipping_mode_Same Day` | 6.06% | 97.49% |
| 5–309 | All remaining features | < 0.42% each | 100.00% |

**Interpretation:** The top 2 features account for **85% of all model learning**. This is not a modelling limitation — it is a genuine business discovery. Delivery outcomes are determined almost entirely by the **logistics decision made at dispatch** (which mode, how quickly), not by customer demographics, product category, or financial profile. This finding gives SwiftChain a precise, actionable lever: change the shipping mode decision, and delivery outcomes change predictably.

---

## 💡 Business Insights & Recommendations

### Finding 1 — Standard Class is the Dominant Risk Factor

| Shipping Mode | Late Rate | On-Time Rate | Early Rate |
|---------------|:---------:|:------------:|:----------:|
| **Standard Class** | **38.1%** | 21.3% | 40.6% |
| Same Day | 4.2% | 42.7% | 53.1% |
| First Class | 1.3% | 0.2% | **98.5%** |
| **Second Class** | **0.3%** | 23.0% | **76.8%** |

Standard Class carries a **38.1% late rate** — nearly 2× the global average of 22.8%. Shifting just 10% of Standard Class orders to Second Class (~910 orders per period) would prevent approximately **347 late deliveries per period**.

### Finding 2 — Market is Not a Meaningful Predictor

All five markets fall within ±1.7 pp of the 22.8% global average. Geographic market provides no additional discriminating power once shipping mode is known.

### Finding 3 — Dispatch Lag is the Second Lever

Orders dispatched within 3 days carry significantly lower late-delivery risk. `shipping_duration` accounts for 17.94% of model learning — the second strongest predictor.

### Prioritised Recommendations

| Priority | Recommendation |
|----------|---------------|
| 🔴 HIGH | Implement a routing policy that auto-flags Standard Class orders for mode-upgrade review, especially for high-value customers |
| 🔴 HIGH | Enforce a 3-day maximum dispatch SLA; auto-escalate orders exceeding this threshold |
| 🟡 MEDIUM | Deploy the model's Late prediction (68.1% recall) as a real-time trigger for automated customer risk notifications |
| 🟡 MEDIUM | Redirect customer satisfaction resources from market-based segmentation to mode-based risk monitoring |
| 🟢 STANDARD | Engineer mode × dispatch_lag interaction features in the next model iteration to improve the On-Time class (currently Recall = 7.4%) |

---

## 🛠 Tech Stack

| Category | Tool |
|----------|------|
| Language | Python 3.x |
| Data manipulation | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine learning | Scikit-learn (GradientBoostingClassifier) |
| Model serialisation | Joblib |
| Web application | Streamlit |
| Hosting | Streamlit Community Cloud |
| Version control | Git / GitHub |
| Development environment | Google Colab |

---

## 📁 Repository Structure

```
swiftchain-delivery-prediction/
│
├── SwiftChain_Delivery_Delay_Prediction_Portfolio.ipynb   ← Full analysis notebook
├── app.py                                                  ← Streamlit web application
├── swiftchain_delay_predictor.pkl                          ← Trained model (serialised)
├── swiftchain_scaler.pkl                                   ← Fitted StandardScaler
├── requirements.txt                                        ← Python dependencies
├── .gitignore                                              ← Files excluded from git
└── README.md                                               ← This file
```

---

## 🚀 Running the App Locally

```bash
# 1. Clone the repository
git clone https://github.com/cssadewale/swiftchain-delivery-prediction.git
cd swiftchain-delivery-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. Python 3.8+ required.

---

## 👤 Author

**Adewale Samson Adeagbo**
Lead Data Scientist / ML Engineer · Lagos, Nigeria
*3MTT Cohort 3 Fellow — DSML Track (Darey.io) · Director & Data Lead, HMG Concepts*

| | |
|---|---|
| 📧 **Email** | buildingmyictcareer@gmail.com |
| 📱 **Phone** | +2348100866322 |
| 💼 **LinkedIn** | [linkedin.com/in/adewalesamsonadeagbo](https://linkedin.com/in/adewalesamsonadeagbo) |
| 🐙 **GitHub** | [github.com/cssadewale](https://github.com/cssadewale) |
| 🌐 **Portfolio** | [cssadewale.pages.dev](https://cssadewale.pages.dev) |

---

## 🏛 Programme Attribution

| | |
|---|---|
| **Funding Body** | Federal Ministry of Communications, Innovation & Digital Economy, Federal Republic of Nigeria |
| **Executing Agency** | National Information Technology Development Agency (NITDA) |
| **Training Provider** | [Darey.io](https://darey.io) — Official 3MTT Partner |
| **Programme** | 3MTT DSML Track — Data Science & Machine Learning |
| **Cohort** | Cohort 3 |
| **Project Number** | 4 of 4 (Final Capstone) |
| **Programme Website** | [3mtt.nitda.gov.ng](https://3mtt.nitda.gov.ng) |

---

*Part of a 4-project supervised ML portfolio — multiclass classification, binary classification, and regression — each with a live Streamlit deployment. See all projects at [cssadewale.pages.dev](https://cssadewale.pages.dev).*
