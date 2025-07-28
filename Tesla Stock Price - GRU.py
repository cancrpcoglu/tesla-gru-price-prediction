import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')  # Hataları yok sayar

import os  # Dosya yolunu bulmamızı sağlar

# Dosya yolunu bulup içinde gezinir
for dirname, _, filenames in os.walk("Users\\RSA-Kadir\\Desktop\\CAN\\Can's Projects"):
    for filename in filenames:
        print(os.path.join(dirname, filename))  # İki yolu birleştirip tek yol oluşturur

data = pd.read_csv('TSLA.csv')  # TSLA.csv'sini okur ve açar
test_data = data.tail(90)
data = data.iloc[:-90]
length_data = len(data)  # Toplam satır sayısını hesaplar
length_train = round(length_data)  # Eğitim için ayrılacak veri sayısını ayarlar

train_data = data[:length_train].iloc[:, :2]  # Seçilen satırlardan ilk 2 sütunu alır
train_data['Date'] = pd.to_datetime(train_data['Date'])  # Tarih ve saat verilerini kolayca işler

dataset_train = train_data.Open.values  # 'Open' değerlerini 'dataset_train' e kaydeder
print(dataset_train.shape)  # Şeklini düzene sokar
dataset_train = np.reshape(dataset_train, (-1, 1))  # Boyutunu düzenler
print(dataset_train.shape)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))  # Min ve max değerleri 0 ve 1 arasında ayarlar
dataset_train_scaled = scaler.fit_transform(dataset_train)  # Verileri 0-1 arasında ölçekler
dataset_train_scaled.shape

X_train = []
y_train = []
time_step = 50  # Kaç adımda bir yazılacağını gösterir

for i in range(time_step, length_train):
    X_train.append(dataset_train_scaled[i - time_step:i, 0])  # 'time_step' kadar geçmiş zaman adımını ekler
    y_train.append(dataset_train_scaled[i, 0])  # Şu anki zaman adımındaki değeri ekler
X_train, y_train = np.array(X_train), np.array(y_train)
print("Shape of X_train before reshape :", X_train.shape)
print("Shape of y_train before reshape :", y_train.shape)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # Boyut değiştirme
y_train = np.reshape(y_train, (y_train.shape[0], 1))

from keras.models import Sequential
from keras.layers import Dense, GRU

model_gru = Sequential()
model_gru.add(GRU(100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model_gru.add(GRU(100, return_sequences=False))
model_gru.add(Dense(64))
model_gru.add(Dense(1))
model_gru.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
history = model_gru.fit(X_train, y_train, epochs=100, batch_size=32)

# Sonuçları döngü ile tahmin et
predictions = []
X_input = dataset_train_scaled[-time_step:]  # Son 50 günün verisini al
X_input = np.reshape(X_input, (1, time_step, 1))

for _ in range(90):  # 90 gün tahmin yapma
    next_pred = model_gru.predict(X_input)
    predictions.append(next_pred[0, 0])
    # Yeni tahmini ekle ve eski verileri kaydır
    X_input = np.append(X_input[:, 1:, :], np.reshape(next_pred, (1, 1, 1)), axis=1)

# Tahminleri ölçekten çıkar
future_predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Tarihleri oluştur
last_date = train_data['Date'].iloc[-1]
future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 91)]

# Gelecek tahmin verisini oluştur
future_data = pd.DataFrame({'Date': future_dates, 'Predicted_Open':future_predictions.flatten()})  # Flatten, DataFrame'e uygun hale getirir

# Tahminleri CSV dosyasına kaydet
future_data.to_csv('future_predictions.csv', index=False)

# Tahmin sonuçlarını görselleştir
plt.figure(figsize=(30, 10))
plt.plot(future_predictions, color="b", label="Predicted")
plt.xlabel("Days")
plt.ylabel("Open price")
plt.title("GRU Model Future Predictions")
plt.legend()
plt.show()

print("Tahminler kaydedildi.")
