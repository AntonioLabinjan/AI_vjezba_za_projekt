# google colab obriše ove csv datasetove kad isteče session, pa ih je najbolje uploadat nanovo svaki put 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Učitani podaci
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Ovo ni bitno, samo pogledamo koji su podaci
print(train_data.head())
print(test_data.head())

# I dalje nebitno...statistika
print(train_data.info())
print(train_data.describe())

# Dali nan fale neke vrijednosti
print(train_data.isnull().sum())
print(test_data.isnull().sum())

# Pretpostavimo da niš strašno ne fali

# Skaliranje
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_data.drop('class', axis=1))
X_test_scaled = scaler.transform(test_data)

# Labele
y_train = train_data['class']

# train/test
X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# definiranje modela
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')  # Assuming 10 classes
])

# kompajliranje
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# treniranje
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# validacija
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy:.2f}")

# Ovo ispod se zakomentira kad samo testiramo
# test sa pravima podacima
test_predictions = model.predict(X_test_scaled)
test_classes = np.argmax(test_predictions, axis=1)

# dataframe s predikcijama
predictions_df = pd.DataFrame({
    "ID": list(range(len(test_classes))),
    "class": test_classes
})

# spremimo u csv
predictions_df.to_csv('submission.csv', index=False)  

