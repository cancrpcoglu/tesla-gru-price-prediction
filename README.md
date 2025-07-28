# tesla-gru-price-prediction
Tesla stock price prediction using GRU

1-) This project uses historical stock data of Tesla (TSLA) to predict future prices using an GRU (Gated Recurrent Unit) deep learning model

2-) About the Project
Dataset: TSLA.csv

3-) Goal: Predict future stock prices by learning temporal patterns in the data

Model Used: Gated Recurrent Unit (GRU)

4-) Libraries Used
-pandas
-numpy
-matplotlib
-scikit-learn
-tensorflow / keras

5-) How to Run

  1-) Clone the repository:

  git clone https://github.com/yourusername/tesla-gru-stock-prediction.git
  cd tesla-gru-stock-prediction

  2-) Install the required libraries:


  pip install -r requirements.txt
  3-) Open and run gru_model.ipynb in Jupyter Notebook.

6-)Notes
The dataset was normalized before training and inverse-transformed after prediction.

The model is trained on historical closing prices only.

Time-based train/test split was applied to preserve the sequence of the data.

7-)Author
Can Çorapçıoğlu — Final year Computer Engineering student at Atılım University
E-Mail: cancorapcioglu1@gmail.com

LinkedIn: [LinkedIn](https://www.linkedin.com/in/can-%C3%A7orap%C3%A7%C4%B1o%C4%9Flu-15a340247/)

This model is based on features in the TSLA.csv dataset, with labels for open values and close values used for 70% training and 30% validation. After training, the model predicts prices for the next three months.
