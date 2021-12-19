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
    """Pré-traite les données d'entrées (csv) et retourne les deux bases de test et d'entrainement.

    Returns:
        tableau d'entiers : Base d'entrainement (Tirages) => x_train
        tableau d'entiers : Base d'entrainement (Labels) => y_train
        tableau d'entiers : Base de test (Tirages) => x_test
        tableau d'entiers : Base de test (Labels) => y_test
    """
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
    """
    Initialise le modèle grâce à partir des bases d'entrainement et de test et le sauvegarde
    """
    x_train,x_test,y_train,y_test = return_data()

    mvs = svm.SVC(probability=True, C=0.001, gamma=0.0001, kernel='linear')
    modele = mvs.fit(x_train,y_train)
    
    save_model(modele)


def random_tirage():
    """Retourne un tirage valide.

    Returns:
        tableau d'entiers : Taille 7 => 5 nombres entre 1 et 50 (différents)
                                        + 2 nombres étoiles entre 1 et 12 (différents)
                            Exemple : [10,21,34,4,19,6,7] 
    """
    tirage = []
    arr50 = np.arange(1,51)
    arr50 = list(arr50)
    arr12 = np.arange(1,13)
    arr12 = list(arr12)
    for _ in range(5):
        # random entre 0 et 49
        idx = np.random.randint(len(arr50))
        
        # choisit ce nombre dans le tableau
        tirage.append(arr50[idx])

        # enlève ce nombre du tableau
        arr50.pop(idx)

    for _ in range(2):
        # random entre 0 et 11
        idx = np.random.randint(len(arr12))

        # choisit ce nombre dans le tableau
        tirage.append(arr12[idx])

        # enlève ce nombre du tableau
        arr12.pop(idx)
    return tirage

def metrics_model():
    """Retourne les métriques pertinentes concernant le modèle.

    Returns:
        string : JSON contenant les métriques du modèle
    """

    # chargement du modèle
    modele = load_model()

    # Retourne les bases de test et d'entrainement
    x_train,x_test,y_train,y_test = return_data()

    # Prédit les valeurs à l'aide du modèle
    y_pred = modele.predict(x_test)

    # Retourne les métriques du modèle
    return metrics.classification_report(y_test, y_pred)

def save_model(modele):
    """
    Sauvegarde le modèle dans le fichier "modele_pickle.pkl"
    """

    with open("modele_pickle.pkl","wb") as f:
        pickle.dump(modele, f)

def load_model():
    """
    Charge le modèle depuis le fichier "modele_pickle.pkl"
    """

    with open("modele_pickle.pkl","rb") as f:
        modele = pickle.load(f)
    return modele
    
def get_bon_tirage():
    """Génère des tirages valides jusqu'à ce que la prédiction que le tirage actuel
    est gagnant soit suffisamment élevé (> 6.2%)

    Returns:
        tableau d'entiers : Tirage valide considéré comme ayant de "fortes" chances de gagner
    """

    # Chargement du modèle
    modele = load_model() 
    while True:
        # Tirage valide
        x = random_tirage()

        # Prédiction de ce tirage
        predi_test = modele.predict_proba([x])
        if (predi_test[0][1] > 0.062):
            return x

# On initialise le modèle s'il n'est pas sauvegardé via pickle dans le répertoire courant
if not(os.path.exists("modele_pickle.pkl")):
    init_modele()
