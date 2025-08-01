# Tesla Stock Price Prediction using GRU

## 📌 Project Overview

This project utilizes a Gated Recurrent Unit (GRU)-based deep learning model to forecast Tesla Inc. stock prices based on historical time-series data. GRU is selected due to its efficiency and performance advantages over traditional RNNs when working with sequential data.

## 🧰 Technologies Used

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib
- Scikit-learn

## 🗂️ Dataset

- Source: [Tesla stock historical data](https://www.kaggle.com/code/serkanp/tesla-stock-price-prediction)
- Features: `Open`, `High`, `Low`, `Close`, `Volume`
- Train-Test Split: 70% training, 30% testing

## 🧠 Model Architecture

- GRU-based recurrent model with several dense layers
- Applied hyperparameter tuning (epochs, learning rate, batch size)
- Used dropout layers to reduce overfitting
- Evaluation metrics: Loss (MSE), Accuracy (if applicable)

## 📊 Results

- Plots include loss graph and predicted vs actual closing prices

## 🚀 How to Run
pip install -r requirements.txt
python gru_model.py

## ✍️ Author
Can Çorapçıoğlu
[GitHub](https://github.com/cancrpcoglu) | [LinkedIn](https://www.linkedin.com/in/can-%C3%A7orap%C3%A7%C4%B1o%C4%9Flu-15a340247/)
