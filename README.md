# 🛣️ Pavement Maintenance Prediction App

This Streamlit app predicts whether a road segment needs maintenance based on condition data such as PCI, IRI, rutting, traffic volume (AADT), and environmental factors like rainfall. It also estimates repair costs and simulates ROI if repair is delayed.

Built with:
- Python
- Streamlit
- XGBoost
- Pandas, NumPy, Seaborn, Matplotlib
- Preprocessed from a dataset hosted on Kaggle

---

## 🔍 Features

- Input road condition manually through UI
- Predicts `Needs Maintenance` (binary classifier)
- Estimates repair cost and cost efficiency
- Simulates ROI over time
- Shows impact of delayed maintenance
- Clean visual charts (cost vs ROI over time)

---

## 📦 Setup

Clone this repo:

```bash
git clone https://github.com/yourusername/pavement-maintenance-app.git
cd pavement-maintenance-app
```
## Install the dependencies:
```bash
pip install -r requirements.txt
```

If you're using kagglehub, make sure you’ve authenticated via your Kaggle token (set your environment variable KAGGLE_USERNAME and KAGGLE_KEY, or place kaggle.json properly).

## 🚀 Run the App
```bash
streamlit run app.py
```
The app will open in your default browser.

## 📁 Folder Structure

├── app.py                  
├── requirements.txt        
├── model_xgb.pkl           
└── README.md

## 📊 Dataset
Dataset is fetched from:
Kaggle - Pavement Dataset by gifreysulay

The app uses kagglehub to download the dataset automatically.

## ⚙️ How It Works

1. User enters road details (PCI, IRI, Rutting, Rainfall, etc.)
2. The app engineers features like:
   - Years since last maintenance
   - Road severity factor
   - PCI class, traffic level, rainfall category
   - Road cost estimation based on type & material
3. One-hot & ordinal encoding are applied
4. XGBoost model predicts if maintenance is needed
5. If yes, the app calculates:
   - Estimated repair cost
   - Cost efficiency score
   - ROI simulation over time
   - Cost/ROI impact from repair delays
  
## 🧠 Model Info

 - Classifier: XGBClassifier
 - Objective: Binary classification — Predict if Needs Maintenance == 1
 - Evaluation: logloss
 - Input: 25+ engineered features
 - Feature importance is not shown but can be visualized separately

## ⚠️ Disclaimer
This is a prototype/demo for road asset management use cases.
Cost estimation, ROI projections, and maintenance predictions are based on simplified assumptions and simulated values. Do not use for real-world financial decisions without expert validation.

## 📬 Contact
Made by [Linkedin](http://linkedin.com/in/muh-amri-sidiq)
Feel free to open an issue or pull request if you'd like to collaborate.
