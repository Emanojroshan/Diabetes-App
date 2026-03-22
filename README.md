# 🩺 Diabetes Prediction Web App

A complete end-to-end Machine Learning web application that predicts whether a person is diabetic or not based on medical input features. The project demonstrates the full ML pipeline including data preprocessing, model training, evaluation, and deployment using a web interface.

---

## 🚀 Features

* Predicts diabetes using trained Machine Learning model
* Clean and modular project structure
* End-to-end ML pipeline (training → saving → deployment)
* Uses real-world dataset (PIMA Indians Diabetes Dataset)
* Scalable and reusable code

---

## 📊 Input Features

The model takes the following medical parameters as input:

* Pregnancies
* Glucose Level
* Blood Pressure
* Skin Thickness
* Insulin Level
* Body Mass Index (BMI)
* Diabetes Pedigree Function
* Age

---

## 🧠 Model Details

* Algorithm: **RandomForestClassifier**
* Data Preprocessing: **StandardScaler**
* Train-Test Split: **80% / 20%**
* Evaluation Metric: **Accuracy Score**

---

## 📈 Model Performance

* Accuracy: **~75% (approx)**
* Balanced dataset split using stratification
* Scaler applied to avoid data leakage

---

## 🗂️ Project Structure

```
Diabetes-App/
│── app/
│   └── app.py                # Web application (Flask)
│
│── model/
│   ├── model.pkl            # Trained ML model
│   └── scaler.pkl           # Saved scaler
│
│── data/
│   └── diabetes.csv         # Dataset
│
│── notebooks/
│   └── training.py          # Model training script
│
│── README.md
│── requirements.txt
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Emanojroshan/Diabetes-App.git
cd Diabetes-App
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

```bash
cd notebooks
python training.py
```

This will:

* Train the model
* Save `model.pkl` and `scaler.pkl` inside the `model/` folder

---

## ▶️ Run the Application

```bash
cd app
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:5000/
```

---

## 🖥️ Application Preview

*Add your app screenshot here*

```
![App Screenshot](screenshot.png)
```

---

## 🔮 Future Improvements

* Add multiple model comparison (Logistic Regression, SVM, etc.)
* Improve UI/UX design
* Deploy application on cloud (Render / Streamlit Cloud)
* Add model explainability (SHAP)
* Add input validation and error handling

---

## 👨‍💻 Author

**Emanojroshan**

* GitHub: https://github.com/Emanojroshan
* LinkedIn: (Add your LinkedIn link)

---

## ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!
