# MLOps Lab CIE

## 📌 Project Overview

This project demonstrates a complete **MLOps pipeline** for predicting **compound yield (mg)** using machine learning.
It covers model training, API deployment, simulation of requests, monitoring, and model registration.

---

## ⚙️ Steps Performed

### 1. Model Training

* Trained multiple models (Lasso, Random Forest)
* Selected best model based on **MAE**
* Saved best model in `models/`

### 2. API Deployment

* Built REST API using **FastAPI**
* Endpoints:

  * `/status` → Health check
  * `/infer` → Prediction endpoint

### 3. Traffic Simulation

* Generated normal and drifted inputs
* Sent requests to API
* Logged predictions in `logs/predictions.jsonl`

### 4. Monitoring & Drift Detection

* Compared live data with training data
* Detected drift in features
* Stored results in `results/step3_s5.json`

### 5. Model Registration

* Used **MLflow** to register best model
* Stored details in `results/step4_s6.json`

---

## 📂 Project Structure

```
Internals_Basics/
└── MLOPs_Lab_CIE/
    ├── src/
    ├── data/
    ├── models/
    ├── logs/
    ├── results/
```

---

## 📊 Results

The `results/` folder contains:

* `step1_s1.json` → Training results
* `step2_s4.json` → API test results
* `step3_s5.json` → Drift monitoring
* `step4_s6.json` → Model registration

---

## 🚀 How to Run

### 1. Train Model

```
python src/train.py
```

### 2. Start API

```
python -m uvicorn src.api:app --port 8000
```

### 3. Simulate Traffic

```
python src/simulate_traffic.py
```

### 4. Monitor

```
python src/monitor.py
```

### 5. Register Model

```
python src/register_model.py
```

---

## 🛠️ Technologies Used

* Python
* FastAPI
* MLflow
* Scikit-learn
* Pandas

---

## ✅ Conclusion

This project successfully implements an end-to-end **MLOps workflow**, including model development, deployment, monitoring, and lifecycle management.
