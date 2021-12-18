import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("EuroMillions_numbers.csv", sep=";")

y = np.where(data["Winner"]>=1, 1, 0)
y=list(y)

for i in range(5268):
    y.append(0)

def random_tirage():
    tirage = []
    arr50 = np.arange(1,51)
    arr50 = list(arr50)
    arr12 = np.arange(1,13)
    arr12 = list(arr12)
    for _ in range(5):
        idx = np.random.randint(len(arr50))
        tirage.append(arr50[idx])
        arr50.pop(idx)

    for _ in range(2):
        idx = np.random.randint(len(arr12))
        tirage.append(arr12[idx])
        arr12.pop(idx)
    return tirage

x = [[data["N1"][i], data["N2"][i], data["N3"][i], data["N4"][i], data["N5"][i], data["E1"][i], data["E2"][i]] for i in range(data.shape[0])]

# 1317 * 4 = 5268
for i in range(5268):
    x.append(random_tirage())

x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size = 0.25,random_state=0)

mvs = svm.SVC(probability=True)
modele = mvs.fit(x_train,y_train)
y_pred = modele.predict(x_test)


#print(metrics.confusion_matrix(y_test,y_pred))

# A quel point le modèle est correct ?
print("Précision/score du modèle :",metrics.accuracy_score(y_test,y_pred)*100, "%")