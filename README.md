# CIS325-Bank-Telemarketing-ML-Deployment
## Project Overview

This project develops and deploys a machine learning model that predicts whether a bank customer will subscribe to a term deposit based on demographic, economic, and marketing campaign data.

The project follows an end-to-end machine learning workflow including:

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training & Evaluation
- MLflow Experiment Tracking
- Flask REST API Development
- Swagger UI Documentation
- Docker Containerization
- Docker Hub Registry Integration
- Cloud Deployment

---

## Dataset

Dataset: Bank Marketing Dataset

Target Variable:

- y = Term Deposit Subscription
  - 1 = Yes
  - 0 = No

Features include:

- Customer demographics
- Campaign contact information
- Previous marketing outcomes
- Economic indicators

---

## Models Evaluated

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

Final selected model:

**LightGBM Classifier**

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- LightGBM
- MLflow
- Flask
- Flasgger (Swagger UI)
- Docker
- GitHub

---

## API Endpoint

### Predict Subscription

**POST** `/predict`

Returns:

```json
{
  "prediction": 0,
  "probability": 0.237,
  "threshold": 0.4
}
```

---

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run API:

```bash
python app.py
```

Swagger UI:

```text
http://127.0.0.1:5001/apidocs
```

---

## Docker

Build image:

```bash
docker build -t bank-telemarketing-api .
```

Run container:

```bash
docker run -p 5001:5001 bank-telemarketing-api
```

---

## Repository

Source Code:

https://github.com/daymyos/CIS325-Bank-Telemarketing

---

## Author

Dayana Yos

CIS325 – Data Science Final Project
