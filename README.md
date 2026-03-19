# 🚦 Smart Traffic Accident Prediction & Prevention System

## Project Structure
```
traffic_app/
│
├── app.py                        ← Flask backend
├── requirements.txt              ← Python dependencies
├── accident_severity_model.pkl   ← Your trained ML model (copy here)
├── README.md
│
└── templates/
    └── index.html                ← Frontend UI
```

---

## ⚙️ Setup & Run Instructions

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Copy your trained model
Copy the `accident_severity_model.pkl` file (saved from your Colab notebook)
into the `traffic_app/` folder.

### Step 3: Run the Flask app
```bash
python app.py
```

### Step 4: Open in browser
```
http://localhost:5000
```

---

## 🔁 How It Works

```
User fills the form (Frontend - HTML/CSS/JS)
           ↓
Clicks "Predict Accident Severity"
           ↓
JavaScript sends POST request to /predict (Backend - Flask)
           ↓
Flask preprocesses input & feeds to ML model
           ↓
Random Forest predicts severity (Low / Medium / High)
           ↓
Flask returns JSON with prediction + probability + tips
           ↓
Frontend displays result with risk alert & prevention tips
```

---

## 🧠 ML Model Info
- **Algorithm:** Random Forest Classifier
- **Target:** Accident Severity (0=Low, 1=Medium, 2=High)
- **Dataset:** accident_prediction_india.csv (3000 rows, 22 features)
- **Key Features:** Weather, Road Type, Lighting, Alcohol, Speed Limit, Driver Age, etc.

---

## 📦 Tech Stack
| Layer    | Technology        |
|----------|-------------------|
| Frontend | HTML, CSS, JS     |
| Backend  | Python + Flask    |
| ML Model | Scikit-learn (Random Forest) |
| Model Persistence | Joblib |
