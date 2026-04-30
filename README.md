# Delhi AQI Prediction Model

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=flat&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

An end-to-end machine learning project that predicts Delhi's Air Quality Index (AQI) using XGBoost regression, achieving an R² score of 0.93+.

---

## 📌 Problem Statement
Air pollution is a critical public health issue in Delhi. This project builds a predictive model to forecast AQI levels based on key pollutant concentrations, enabling proactive air quality management.

## 📊 Dataset
- Source: Historical Delhi AQI data
- Features: PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene
- Target: AQI value

## 🔍 Approach
1. **Exploratory Data Analysis** — identified key pollutant correlations and seasonal trends
2. **Feature Engineering** — handled missing values, scaled features, created interaction terms
3. **Modelling** — trained and compared multiple regression models
4. **Hyperparameter Tuning** — used GridSearchCV to resolve overfitting and optimise XGBoost
5. **Evaluation** — assessed performance using R², MAE, and RMSE

## 📈 Results
| Metric | Score |
|--------|-------|
| R² Score | 0.93+ |
| Model | XGBoost Regressor |

## 🚀 How to Run
```bash
# Clone the repo
git clone https://github.com/Sh10bh/DELHI-AQI-PREDICTION-MODEL.git
cd DELHI-AQI-PREDICTION-MODEL

# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook DELHI_AQI_MODEL.ipynb
```

## 🛠 Tech Stack
- **Language:** Python
- **ML:** XGBoost, Scikit-learn
- **Data:** Pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn
