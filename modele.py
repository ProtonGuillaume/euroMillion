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
import pickle
import os

def return_data():
    data = pd.read_csv("EuroMillions_numbers.csv", sep=";")

    y = np.where(data["Winner"]>=1, 1, 0)
    y=list(y)

    for i in range(5268):
        y.append(0)

    x = [[data["N1"][i], data["N2"][i], data["N3"][i], data["N4"][i], data["N5"][i], data["E1"][i], data["E2"][i]] for i in range(data.shape[0])]

    # 1317 * 4 = 5268
    for i in range(5268):
        x.append(random_tirage())

    return model_selection.train_test_split(x,y,test_size = 0.25,random_state=0)


def init_modele():

    x_train,x_test,y_train,y_test = return_data()

    mvs = svm.SVC(probability=True, C=0.001, gamma=0.0001, kernel='linear')
    modele = mvs.fit(x_train,y_train)
    
    save_model(modele)

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

def metrics_model():
    modele = load_model()
    x_train,x_test,y_train,y_test = return_data()

    y_pred = modele.predict(x_test)
    return metrics.classification_report(y_test, y_pred)

def save_model(modele):
    with open("modele_pickle.pkl","wb") as f:
        pickle.dump(modele, f)

def load_model():
    with open("modele_pickle.pkl","rb") as f:
        modele = pickle.load(f)
    return modele
    
if not(os.path.exists("modele_pickle.pkl")):
    init_modele()


def get_bon_tirage():
    modele = load_model() 
    while True:
        x = random_tirage()
        predi_test = modele.predict_proba([x])
        if (predi_test[0][1] > 0.062):
            return x