# 🚀 Predictive Maintenance MLOps Project

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/FastAPI-Production-green?style=for-the-badge&logo=fastapi">
  <img src="https://img.shields.io/badge/MongoDB-Database-brightgreen?style=for-the-badge&logo=mongodb">
  <img src="https://img.shields.io/badge/Scikit--Learn-MachineLearning-orange?style=for-the-badge&logo=scikitlearn">
  <img src="https://img.shields.io/badge/MLOps-End%20to%20End-red?style=for-the-badge">
</p>

---

## 📌 Project Overview

This project is a complete **End-to-End MLOps Pipeline** for predicting machine failures in industrial environments.

The system leverages Machine Learning and MLOps best practices to automate:

- Data Ingestion
- Data Validation
- Data Transformation
- Model Training
- Model Evaluation
- Model Registry
- Prediction Service
- Deployment Pipeline

The objective is to predict whether a machine is likely to fail based on operational parameters such as temperature, rotational speed, torque, tool wear, and machine type.

---

# 🎯 Business Problem

Unexpected machine failures can result in:

- Production downtime
- Increased maintenance costs
- Revenue loss
- Reduced operational efficiency

This solution enables proactive maintenance by predicting failures before they occur.

---

# 🏗️ Project Architecture

```text
MongoDB
   │
   ▼
Data Ingestion
   │
   ▼
Data Validation
   │
   ▼
Data Transformation
   │
   ▼
Model Training
   │
   ▼
Model Evaluation
   │
   ▼
Model Registry
   │
   ▼
Prediction Pipeline
   │
   ▼
FastAPI Web Application
```

---

# ⚙️ Tech Stack

## Programming

- Python

## Machine Learning

- Scikit-Learn
- Random Forest Classifier
- NumPy
- Pandas

## Backend

- FastAPI
- Uvicorn

## Database

- MongoDB Atlas

## MLOps

- Model Registry
- Pipeline Automation
- Artifact Management
- GitHub Actions
- CI/CD Workflow

## Deployment

- Vercel / Render Ready
- Docker Ready

---

# 📊 Dataset Features

| Feature | Description |
|----------|-------------|
| Air temperature [K] | Ambient air temperature |
| Process temperature [K] | Process temperature |
| Rotational speed [rpm] | Machine rotational speed |
| Torque [Nm] | Torque value |
| Tool wear [min] | Tool wear duration |
| Type | Machine Type |
| Machine failure | Target Variable |

---

# 🔄 End-to-End Pipeline

## 1️⃣ Data Ingestion

- Connects to MongoDB Atlas
- Fetches raw machine data
- Stores dataset into feature store

### Output

```text
artifact/
└── data_ingestion/
```

---

## 2️⃣ Data Validation

Validates:

- Schema consistency
- Missing columns
- Data integrity
- Dataset quality

### Output

```text
artifact/
└── data_validation/
```

---

## 3️⃣ Data Transformation

Performs:

- Feature selection
- Encoding
- Data preprocessing

### Output

```text
artifact/
└── data_transformation/
```

---

## 4️⃣ Model Training

Algorithm Used:

```python
RandomForestClassifier
```

Training includes:

- Hyperparameter configuration
- Performance evaluation
- Model serialization

### Output

```text
predictive_maintenance_model.pkl
```

---

## 5️⃣ Model Evaluation

Compares:

- Current Production Model
- Newly Trained Model

Acceptance Criteria:

```text
New Model F1 Score > Production Model F1 Score
```

Only better models are promoted.

---

## 6️⃣ Model Registry

Stores approved models for production use.

### Registry Structure

```text
model_bucket/
└── predictive_maintenance_model.pkl
```

---

## 7️⃣ Prediction Pipeline

Loads:

- Production Model
- Preprocessing Object

Generates real-time predictions through FastAPI.

---

# 🌐 FastAPI Application

Users can enter:

- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear
- Machine Type

### Prediction Results

```text
Machine Failure
```

or

```text
No Machine Failure
```

---

# 📁 Project Structure

```text
├── artifact/
├── config/
├── notebook/
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   │
│   ├── pipeline/
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   │
│   ├── entity/
│   ├── configuration/
│   ├── utils/
│   └── logger/
│
├── static/
├── templates/
├── app.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

# 🚀 Running The Project

## Clone Repository

```bash
git clone <your-repository-url>
```

```bash
cd ml-predictive-maintenance
```

---

## Create Environment

```bash
conda create -n mlops python=3.12 -y
```

```bash
conda activate mlops
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Start Training Pipeline

```bash
python main.py
```

---

## Run FastAPI Application

```bash
python app.py
```

---

# 📈 Model Performance

Metrics Used:

- Accuracy
- Precision
- Recall
- F1 Score

Primary Metric:

```text
F1 Score
```

---

# 🔥 Key MLOps Features

✅ End-to-End Training Pipeline

✅ Modular Code Architecture

✅ MongoDB Data Source

✅ Model Registry

✅ Automated Evaluation

✅ Production Model Comparison

✅ Artifact Tracking

✅ FastAPI Deployment

✅ Docker Ready

✅ CI/CD Ready

---

# 🎓 Skills Demonstrated

### Machine Learning

- Classification
- Feature Engineering
- Model Evaluation
- Random Forest

### MLOps

- Pipeline Development
- Model Versioning
- Artifact Management
- Deployment Automation

### Software Engineering

- Object-Oriented Programming
- Exception Handling
- Logging
- Modular Architecture

### Backend Development

- FastAPI
- REST APIs
- Template Rendering

---

# 👨‍💻 Author

### Rohit Rane

Aspiring Machine Learning Engineer | MLOps Enthusiast

- Machine Learning
- MLOps
- FastAPI
- MongoDB
- Python

📧 Add your Email

🔗 Add your LinkedIn

🔗 Add your GitHub

---

## ⭐ If you found this project useful, consider giving it a star.