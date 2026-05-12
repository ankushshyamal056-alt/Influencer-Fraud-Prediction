# 🕵️ Influencer Fraud Detection

> An unsupervised machine learning pipeline that assigns a **fraud risk score (0–100)** to social media influencer profiles, helping brands and agencies identify fake or bot-inflated accounts before partnering with them.

---

## 📌 Project Description

Influencer marketing fraud costs brands billions annually — fake followers, bot-driven engagement, and inflated metrics make it difficult to identify genuine creators. This project builds an **end-to-end fraud detection pipeline** using unsupervised anomaly detection on Instagram/TikTok profile data, requiring **no labeled data to run** while using a small hand-labeled set purely for validation and threshold calibration.

The pipeline assigns every profile a continuous **fraud risk score** and routes it into an actionable **risk tier** (Low → Critical), giving marketing teams a clear, explainable signal rather than a black-box binary prediction.

---

## 🧠 How It Works

Because the vast majority of real-world influencer data is **unlabeled**, the pipeline relies on two complementary unsupervised anomaly detectors:

| Model | Mechanism |
|---|---|
| **Isolation Forest** | Isolates anomalies using random feature splits — outliers need fewer splits to isolate |
| **Local Outlier Factor (LOF)** | Detects anomalies by comparing local density to k-nearest neighbours |

The final fraud score is a **weighted ensemble**:

```
fraud_score = 0.6 × Isolation Forest score + 0.4 × LOF score
```

A small hand-labeled set of 200 profiles (100 genuine, 100 fraud) is used **only** for threshold calibration and held-out validation via stratified 5-fold cross-validation.

---

## 📊 Dataset

- **5,000** Instagram/TikTok profile rows with pre-computed engagement metrics
- **200 labeled profiles** (100 genuine, 100 fraud) for validation only
- **4,800 unlabeled profiles** processed by the unsupervised pipeline

Key raw features: `followers`, `likes`, `comments`, `shares`, `engagement_rate`, `growth rate`

---

## ⚙️ Feature Engineering

Six interaction features are engineered on top of the raw metrics:

| Feature | Formula | Rationale |
|---|---|---|
| `likes_per_follower` | likes / followers | Genuine accounts have consistent per-follower engagement |
| `comments_per_follower` | comments / followers | Bought followers rarely comment |
| `shares_per_follower` | shares / followers | Shares require real intent — strong authenticity signal |
| `comment_to_like_ratio` | comments / likes | Bots inflate likes but not comments |
| `like_to_follower_ratio` | likes / followers | Holistic per-follower like signal |
| `log_followers` | log1p(followers) | Compresses heavy right skew for better model geometry |

---

## 🏷️ Risk Tier System

Each profile is automatically assigned a risk tier and a recommended vetting action:

| Score Range | Tier | Recommended Action |
|---|---|---|
| 0 – 29 | 🟢 Low | Auto-approve |
| 30 – 49 | 🟡 Medium | Manual review required |
| 50 – 79 | 🟠 High | Request audience proof |
| 80 – 100 | 🔴 Critical | Reject / Blacklist |

---

## 🔍 Explainability

The pipeline goes beyond scoring to explain *why* a profile is flagged:

- **SHAP values** (via `TreeExplainer`) reveal each feature's contribution to the Isolation Forest anomaly score
- **LOF vs. Isolation Forest score scatter** highlights ambiguous profiles where the two models disagree — useful for prioritising manual review
- **PCA visualisation** projects the 7-feature space into 2D for cluster inspection

---

## 📁 Project Structure

```
├── Influencer_fraud_detection_improved.ipynb   # Main pipeline notebook
├── Profiles.csv                                # Input dataset
├── Influencer_fraud_detection_full_output.csv  # Output with scores & risk tiers
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap
```

### Run the Notebook

1. Clone this repository
2. Place `Profiles.csv` in the project root
3. Open and run `Influencer_fraud_detection_improved.ipynb` top to bottom

The final output CSV (`Influencer_fraud_detection_full_output.csv`) will be saved automatically.

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data loading and manipulation |
| `matplotlib`, `seaborn` | Visualisation and EDA |
| `scikit-learn` | Isolation Forest, LOF, PCA, cross-validation, metrics |
| `shap` | Model explainability (SHAP values) |

---

## 📈 Validation Results

The pipeline is evaluated on the 200-profile labeled subset using **Stratified 5-Fold Cross-Validation**. The optimal fraud score threshold is selected by maximising mean F1 across folds. A full **Precision-Recall curve** is also plotted for the labeled set.

---

## 🤝 Use Cases

- **Brand safety teams** vetting influencers before campaign spend
- **Influencer marketing platforms** building automated fraud filters
- **Agencies** adding a quantitative fraud layer to influencer shortlisting
- **Researchers** studying engagement anomaly detection at scale

---


