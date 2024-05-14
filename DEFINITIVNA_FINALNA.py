# ispod ovega je novo
from sklearn.utils.class_weight import compute_class_weight

import pandas as pd
import numpy as np



#učitavanje datoteka
data_set = pd.read_csv("train_data.csv")
test_set = pd.read_csv("test_data.csv")


#pregled datoteka za null vrijednosti
#print(data_set.isnull().sum())
#print(test_set.isnull().sum())

#pregled podataka
type(data_set)
data_set.columns
df = data_set.describe()
data_set.shape
data_set.size

#uspoređivali smo stupce train i test data kako bi vidjeli koje stupce trebamo iskoristiti
data_set.columns
test_set.columns

#odvojili smo stupac class od svih ostalih stupaca
data_x = data_set[data_set.columns[:-1]]
data_y = data_set['class']

# ovo je novo
'''
class_weights = compute_class_weight('balanced', np.unique(data_y), data_y)
class_weight_dict = dict(zip(np.unique(data_y), class_weights))
'''
'''
class_weights = compute_class_weight('balanced', np.unique(data_y), data_y)
class_weight_dict = dict(zip(np.unique(data_y), class_weights))
'''
'''
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', np.unique(data_y), data_y)
class_weight_dict = dict(zip(np.unique(data_y), class_weights))
'''
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(data_y), y=data_y)
class_weight_dict = dict(zip(np.unique(data_y), class_weights))

# ovo gore je novo

#standardizacija podataka
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data_x)
data_x_stand = scaler.fit_transform(data_x)
data_x_stand.mean(axis=0).round()
data_y_stand = scaler.fit_transform(test_set)

#normalizacija podataka
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
mm.fit(data_x_stand)
data_x_mm = mm.transform(data_x_stand)
#print(data_x_mm.min(axis=0).round(), data_x_mm.max(axis=0).round())

#nasumična podjela podataka

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_x_stand, data_y, test_size=0.2, random_state=42, shuffle = True)
#print(x_train.shape, x_test.shape)
#print(y_train.shape)

'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_x_mm, data_y, test_size=0.2, random_state=42, shuffle = True)
'''
#standardizacija
scaler.fit(x_train)
x_train_stand = scaler.transform(x_train)
x_test_stand = scaler.transform(x_test)
print(x_train_stand.shape)

#definiranje modela
import tensorflow as tf

'''
prvi funkcionalni model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(x_train_stand.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
    ])
'''

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(x_train_stand.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
    ])

#compile podataka
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# prvo treniranje modela
# history = model.fit(x_train_stand, y_train, epochs=25, batch_size=64, validation_data=(x_test, y_test), shuffle=True)

# drugo treniranje modela
# history = model.fit(x_train_stand, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test), shuffle=True)

# treće treniranje modela
# history = model.fit(x_train_stand, y_train, epochs=37, batch_size=128, validation_data=(x_test, y_test), shuffle=True) 0.89960

# četvrto treniranje modela (najbolje za sad)
#history = model.fit(x_train_stand, y_train, epochs=44, batch_size=128, validation_data=(x_test, y_test), shuffle=True)

# peto treniranje modela
# history = model.fit(x_train_stand, y_train, epochs=48, batch_size=128, validation_data=(x_test, y_test), shuffle=True)

#šesto treniranje modela
# history = model.fit(x_train_stand, y_train, epochs=45, batch_size=128, validation_data=(x_test, y_test), shuffle=True)
# ne prelazimo 44, provajmo s manje

#sedmo treniranje modela
# history = model.fit(x_train_stand, y_train, epochs=43, batch_size=128, validation_data=(x_test, y_test), shuffle=True)

#osmo treniranje modela
#history = model.fit(x_train_stand, y_train, epochs=44, batch_size=64, validation_data=(x_test, y_test), shuffle=True)

#deveto treniranje modela
#history = model.fit(x_train_stand, y_train, epochs=44, batch_size=256, validation_data=(x_test, y_test), shuffle=True)

# deseto treniranje modela
#history = model.fit(x_train_stand, y_train, epochs=44, batch_size=200, validation_data=(x_test, y_test), shuffle=True) #

#jedanaeasto treniranje modela
#history = model.fit(x_train_stand, y_train, epochs=32, batch_size=100, validation_data=(x_test, y_test), shuffle=True, callbacks=[stop]) #


#dvanesto treniranja modela
#history = model.fit(x_train_stand, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test), shuffle=True) #

#trinaesto treniranje modela => dodali smo class weight
# ovo nakon shuffle = True je novo => ovo je dalo 92.920 na kaggleu
history = model.fit(x_train_stand, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test), shuffle=True, class_weight=class_weight_dict) # 0.90420

#history = model.fit(x_train_stand, y_train, epochs=160, batch_size=32, validation_data=(x_test, y_test), shuffle=True, class_weight=class_weight_dict) 

#četrnaesto treniranje modela
#history = model.fit(x_train_stand, y_train, epochs=224, batch_size=32, validation_data=(x_test, y_test), shuffle=True) #0.90520

#petnaeasto treniranje modela
#history = model.fit(x_train_stand, y_train, epochs=256, batch_size=64, validation_data=(x_test, y_test), shuffle=True) #0.8964

# šesnaesto
#history = model.fit(x_train_stand, y_train, epochs=264, batch_size=61, validation_data=(x_test, y_test), shuffle=True) #0.8976

#history = model.fit(x_train_stand, y_train, epochs=270, batch_size=60, validation_data=(x_test, y_test), shuffle=True) # 0.905

#history = model.fit(x_train_stand, y_train, epochs=280, batch_size=60, validation_data=(x_test, y_test), shuffle=True) # 0.89

#Za ovaj batch size je gore ako dižemo epohe

test_predictions = model.predict(data_y_stand)
test_classes = np.argmax(test_predictions, axis=1)

predictions_df = pd.DataFrame({
    "ID": list(range(len(test_classes))),
    "class": test_classes
})

predictions_df.to_csv('submission.csv', index=False)


'''
import matplotlib.pyplot as plt

# Grafikon za loss
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Grafikon za accuracy
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
'''
