# Telecom Customer Churn Prediction – End-to-End ML Pipeline

An **end-to-end machine** learning project to predict customer churn for a telecom company. The pipeline connects to a MongoDB database for data ingestion, preprocesses the data, trains classification models focused on minimizing false positives, and deploys the best model in AWS. A FastAPI-based web app provides a user interface, and GitHub Actions powers the CI/CD pipeline.

**Table of Contents**

Overview

Tech Stack

Project Structure

Pipeline Details

Data Collection

Preprocessing

Model Training

Model Evaluation

Model Deployment

CI/CD with GitHub Actions

Frontend (FastAPI)

Getting Started

Contributing

License

*----------------------------------------------------------------------*
## Overview

The aim of the project is to predict customer who is likely to leave the telecom company based on several other features. Due to the *imbalanced* nature of the dataset, special emphasis is placed on **maximizing recall** and **reducing false positives**, ensuring that churn-prone customers are not overlooked.

*----------------------------------------------------------------------*
## Tech Stack

**Languages:** Python

**Machine Learning Libraries:** scikit-learn, pandas, NumPy, imbalanced-learn, scipy, statsmodels

**Backend & API:** FastAPI

**Database:** MongoDB

**Storage & Deployment:** AWS, AWS S3

**CI/CD:** GitHub Actions

**Others:** Docker (optional), Pydantic

*----------------------------------------------------------------------*

## Project Structure

├── data/                   # Sample data or data loaders
├── src/                    # ML scripts: preprocessing, training, evaluation
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── model.py
│   └── utils.py
├── models/                 # Saved models
├── webapp/                 # FastAPI application
│   ├── main.py
│   └── templates/
├── .github/workflows/      # GitHub Actions workflows
├── requirements.txt
└── README.md

*----------------------------------------------------------------------*

## Pipeline Details

### Data Collection

Customer usage and subscription data is pulled from MongoDB.

### Preprocessing

Cleans missing and inconsistent values.

Applies encoding, normalization, and feature engineering.

Splits data into train/test sets using stratified sampling.

### Model Training

Multiple classifiers tested (e.g., Logistic Regression, Random Forest).

Hyperparameter tuning applied as needed.

### Model Evaluation

Primary Metric: Recall (due to class imbalance).

Goal: Reduce false positives to avoid misclassifying loyal customers.

Best-performing model is selected and saved.

### Model Deployment

Final model is serialized and uploaded to AWS S3.

Ready for inference via API or frontend.

*----------------------------------------------------------------------*

## CI/CD with GitHub Actions

Automated workflows handle:

Code linting & style checks

Unit tests

Model training and evaluation

Deployment to cloud (AWS)

*----------------------------------------------------------------------*

## Frontend (FastAPI)

A simple FastAPI web interface allows telecom customer data to be submitted for real-time churn prediction.

To run locally:

uvicorn webapp.main:app --reload

*----------------------------------------------------------------------*
## Getting Started

- Clone the Repo

- Install Dependencies

- Start the Web App

*----------------------------------------------------------------------*
### Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve this project.

### License

This project is licensed under the MIT License.


