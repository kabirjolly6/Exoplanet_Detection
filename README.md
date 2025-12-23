# 🌌 Exoplanet Detection using Machine Learning

An end-to-end machine learning system for **exoplanet detection** using NASA astronomical datasets, designed with a strong emphasis on **scientific validity, recall-oriented evaluation, and robust generalization**.

The project leverages **LightGBM** along with a **custom refit strategy (FitSearchCV)** to avoid overfitting and select stable models suitable for real-world scientific discovery.

---

## 📌 Problem Statement

Astronomical missions generate massive volumes of telescope data, making manual exoplanet detection slow and impractical at scale.

**Objective:**  
Develop an automated and reliable ML pipeline to classify celestial objects into:
- **Confirmed Exoplanet**
- **Exoplanet Candidate**
- **False Positive**

with special focus on **minimizing missed exoplanet candidates**.

---

## 🚀 Key Contributions

- 🔬 Recall-first evaluation strategy for candidate discovery  
- 🧠 LightGBM-based classification for structured astronomical data  
- 🛡️ Explicit handling of data leakage and feature integrity  
- ⚙️ **FitSearchCV**: custom hyperparameter refit strategy  
- 🌐 Deployed web application for prediction and retraining  
- 📊 Careful dataset selection based on label stability

---

## 📂 Datasets

### 🔹 KOI (Kepler Object of Interest) — *Primary Dataset*
- Clean and well-labeled
- Stable target definitions
- Used for training and testing

### 🔹 K2 Dataset
- High reported accuracy
- Susceptible to data leakage
- Used only for supplementary validation

### 🔹 TESS (TOI) Dataset
- Large and evolving dataset
- Labels under continuous revision
- Not used for supervised training

---

## 🧪 Feature Engineering

- Final feature set: **17 features**
- Preprocessing steps:
  - Removed empty and redundant columns
  - Eliminated single-value categorical features
  - Excluded target-leaking flags
- **False-positive indicators deliberately excluded** to preserve model integrity

---

## 🤖 Model Details

- **Algorithm:** LightGBM (Gradient Boosted Decision Trees)
- **Why LightGBM?**
  - Efficient on large tabular datasets
  - Handles non-linear feature interactions
  - High recall for exoplanet candidates
- **Primary Metric:** Recall (Candidate class)

---

## 🧠 Custom Refit Strategy: FitSearchCV

Instead of refitting the model with the highest validation score, we optimize for stability using:

```text
min ( |train_score - test_score| + (1 - test_score) ) / 2
```
### Benefits:
- Reduces overfitting
- Avoids underfitting
- Encourages generalizable models
- Better suited for scientific discovery tasks

---

## 🌐 Application Features

- 🔍 Real-time single prediction
- 📁 Batch prediction with downloadable results
- 🔄 Online retraining with new datasets
- 🧪 Interactive web interface

---

## 🔮 Future Work
- Automated periodic retraining
- Intelligent false-positive flagging
- Dataset validation pipeline
- Continuous monitoring for model drift


