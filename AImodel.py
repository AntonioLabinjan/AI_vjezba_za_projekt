'''
# AI: python
# import data
# clean data
# split data into training/test sets
# create model
# train model
# evaluate model
# make predictions
# evaluate predictions
# improve model
# retrain model
# reevaluate model
# make predictions
# ...
# numpy, pandas, matplotlib, seaborn, scikit-learn, sklearn, statsmodels, sklearn-pandas, sklearn-pandas-datareader, yfinance, yahoofinancials, yahoofinancials...

# import data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
music_data = pd.read_csv('music.csv')
#print(music_data)
#ne treba nan cleaning jer je sve okej
#inače trebamo maknut null vrijednosti, ali tu ih nema
# razdvajamo ga u input i output


X = music_data.drop(columns=['genre']) # dataset bez stupca genre
print(X)
#print(music_data['genre']) # samo stupac genre iz dataseta
Y = music_data['genre'] # samo stupac genre iz dataseta
print(Y)
# računanje preciznosti
# moramo razdvojit dataset (80 training, 20 testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# X je input data za naš model
# Y je output data za naš model
# Na temelju X-eva ćemo pokušavat dobit Y-e 
# Sad koristimo algoritam za klasifikaciju
# klasifikator za klasifikaciju
model = DecisionTreeClassifier() # imamo model, sad ga moramo istrenirat
model.fit(X_train, Y_train)
# Moremo delat predikcije za one podatke koje nemamo na temelju postojećih sličnih
#predictions = model.predict([[21, 1], [22, 0]) # predviđamo ča slušaju 21 godišnji muškarac i 22 godišnja žena
predictions = model.predict(X_test)
print(predictions) # 21-godišnji muškarac sluša hiphop, a 22-godišnja žena sluša dance (na temelju uzoraka koji smo izabrali))
score = accuracy_score(Y_test, predictions)
print(score) # prvi put je 100% preciznost, ali svaki put kad opet pokrenemo kod, ona je sve manja i manje, jer koristimo samo mali dio dataseta za delat predicije o cijelemu datasetu
# ča je čišći i bolji dataset, to će naš model bit bolji
# tu pada preciznost testa jer imamo mali dataset
'''
'''
# Krećemo opet
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

#music_data = pd.read_csv('music.csv')
#x=music_data.drop(columns=['genre'])
#y=music_data['genre']

#model = DecisionTreeClassifier()
#model.fit(x,y)

#joblib.dump(model, 'music-recommender.joblib')
model = joblib.load('music-recommender.joblib')
predictions = model.predict([[21, 1]])
print(predictions)
#predictions = model.predict([[21, 1]])

#print(predictions)
'''
# vizualizacija
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
Y = music_data['genre']

model = DecisionTreeClassifier()
model.fit(X, Y)

tree.export_graphviz(model, out_file='music-recommender.dot', feature_names=['age', 'gender'], class_names=sorted(Y.unique()), label='all', rounded=True, filled=True )







































