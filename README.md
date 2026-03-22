# Diabetes Prediction App

A machine learning web application that predicts diabetes using a trained RandomForestClassifier.

## Project Structure

```
Diabetes-App/
│── app/
│   └── app.py          # Flask web application
│
│── model/
│   ├── model.pkl       # Trained ML model
│   └── scaler.pkl      # Fitted StandardScaler
│
│── data/
│   └── diabetes.csv    # Dataset
│
│── notebooks/
│   └── training.py     # Model training script
│
│── README.md
│── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
cd notebooks
python training.py
```

## Run the App

```bash
cd app
python app.py
```
