# 🌫️ Air Quality ML

**Air Quality ML** is a machine learning-based project designed to predict air pollution levels, specifically the **Air Quality Index (AQI)**, using environmental features like temperature, humidity, and particulate matter levels. It includes model training, evaluation, and a real-time AQI prediction web interface built with **Streamlit**.

---

## 📘 Overview

This project leverages regression algorithms to estimate AQI from pollutant and environmental data. It showcases how machine learning can support **environmental monitoring** and promote **public health awareness**.

---

## 🚀 Features

- 🔄 Data collection & preprocessing  
- 🧠 Model training and evaluation (Scikit-learn, XGBoost)  
- 📈 AQI trends visualization  
- 🌐 Streamlit web app for live predictions  
- ☁️ Real-time weather data from OpenWeatherMap API  

---

## 🎯 Project Goals

- ✅ Build and compare regression models for AQI prediction  
- ✅ Evaluate using MAE, RMSE, and R² Score  
- ✅ Real-time prediction with an interactive interface  
- ✅ Insightful data visualizations  

---

## 🛠️ Tech Stack

- 💻 Python 3  
- 📦 Scikit-learn, XGBoost  
- 🌐 Streamlit  
- 🌍 OpenWeatherMap API  
- 📊 Pandas, Matplotlib, Seaborn, Plotly  
- 📓 Jupyter Notebooks  

---

## 📂 Folder Structure

air-quality-ml/
├── streamlit_app.py
├── templates/
├── static/
├── datasets/
├── notebooks/
│ └── XGBoost/
├── models/
├── Plot_AQI.py
├── requirements.txt
└── README.md


---

## 🧠 Models Used

- 🔸 XGBoost Regressor  
- 🔸 Random Forest  
- 🔸 Ridge Regression  

---

## 📊 Model Evaluation

| Model              | R² Score | MAE   | RMSE  |
|--------------------|----------|-------|-------|
| 🌲 Random Forest    | 0.85     | 10.3  | 14.6  |
| ⚡ XGBoost          | 0.83     | 11.0  | 15.2  |
| 🔹 Ridge Regression | 0.71     | 13.5  | 18.1  |

---

## 📈 Visualizations

- 📊 **AQI Trends** — via `Plot_AQI.py`  
- 📒 **Model Evaluations** — in Jupyter Notebooks  
- 📍 **Real-Time Prediction** — with Plotly + Streamlit  

---

## 🔮 Future Improvements

- 🏙️ Extend support to more cities and regions  
- 🎨 Improve UI/UX design  
- ☁️ Cloud deployment (Render, Heroku, Streamlit Cloud)  
- 🔗 Integrate additional AQI APIs (e.g., AQICN)  

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🙋‍♂️ Author

**Created by:** Saikiran  
🔗 [GitHub: saikiran-m18](https://github.com/saikiran-m18)
