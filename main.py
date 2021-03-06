from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import os
import datetime
import csv
import modele

# Ici le chemin du ficiher de sauvegarde des tirages de l'EuroMillions
csv_file = "EuroMillions_numbers.csv"

my_model = modele.load_model()

def validate_date(date_text):
    """Vérifie si la date passée en paramètre correspond à un certain format.
    
    Args:
        date_text (str): La date à vérifier
    
    Returns:
        bool: True si la date est au bon format, False le cas échéant.
    
    """
    try:
        datetime.datetime.strptime(date_text, '%Y-%m-%d')
        return(True)
    except ValueError:
        return(False)


def validate_drawing(item):
    """Vérifie si le tirage passé en paramètre a des numéros différents (hors numéros étoiles).
    
    Args:
        item (Item): Le tirage à vérifier.
    
    Returns:
        bool: True si les numéros sont différents, False le cas échéant.
    
    """
    return(item.n1 == item.n2 or item.n1 == item.n3 or item.n1 == item.n4 or item.n1 == item.n5 or
         item.n2 == item.n3 or item.n2 == item.n4 or item.n2 == item.n5 or item.n3 == item.n4 or
         item.n4 == item.n5)

# Classe décrivant un tirage
class Item(BaseModel):
    """Classe décrivant un tirage.
    
    Attributes:
        n1..n5 (int): Les numéros du tirage.
        e1, e2 (int): Les numéros étoiles.
        date (str): La date du tirage, optionnelle.
        winner (int): Le nombre de gagnants avec ces numéros.
        gain (int): Le gain total de ce tirage.
        
    """
    n1: int = Field(..., gt=0, lt=51, description="The number must be between 1 and 50")
    n2: int = Field(..., gt=0, lt=51, description="The number must be between 1 and 50")
    n3: int = Field(..., gt=0, lt=51, description="The number must be between 1 and 50")
    n4: int = Field(..., gt=0, lt=51, description="The number must be between 1 and 50")
    n5: int = Field(..., gt=0, lt=51, description="The number must be between 1 and 50")
    e1: int = Field(..., gt=0, lt=51, description="The joker number must be between 1 and 12")
    e2: int = Field(..., gt=0, lt=51, description="The joker number must be between 1 and 12")
    date: Optional[str] = None
    winner: Optional[int] = Field(..., gt=-1, description="The number of winners must be greater than -1")
    gain: Optional[int] = Field(..., gte=0, description="The gain must be greater or equal to 0")


# Processus principal
app = FastAPI()


@app.post("/api/predict/")
async def evaluate_prediction(item: Item):
    """Evalue le tirage passé en paramètre.
    
    Args:
        item (Item): Le tirage à évaluer.
    
    Returns:
        float: La probabilité d'avoir la combinaison gagnante.

    """
    if (validate_drawing(item)):
        raise HTTPException(status_code=422, detail="Unprocessable Entity : the numbers should be different from each others.")

    if (item.e1 == item.e2):
        raise HTTPException(status_code=422, detail="Unprocessable Entity : the star numbers should be different from each others.")
    
    # evaluation with the model
    my_pred = my_model.predict_proba([[item.n1,item.n2,item.n3,item.n4,item.n5,item.e1,item.e2]])
    probability = my_pred[0][1]

    return({"message": f"Win probability : {probability} %, Lose probability : {1-probability} %"})

@app.get("/api/predict/")
async def get_prediction():
    """Demande au modèle une prédiction de tirage gagnant.

    Returns:
        item (Item): un tirage qui a une probabilité supposée plus élevée d'être gagnant.
    
    """

    #prédiction de tirage gagnant
    bon_tirage = modele.get_bon_tirage()

    return({"bon_tirage": f"Bon tirage : {bon_tirage}"})

@app.get("/api/model")
async def get_model_details():
    """Retourne les détails du modèle utilisé.

    Returns:
        string: JSON décrivant la métrique, l'algorithme et les paramètres utilisés pour le modèle.
    
    """
    metric = modele.metrics_model()
    algorithm = "Classifier - Support Vector Machines"
    training_parameters = "Kernel = Linear ; C = 0.001 ; Gamma = 0.0001"

    return({"metric": metric, "algorithm": algorithm, "training_parameters": training_parameters})

@app.put("/api/model/")
async def add_item(item: Item):
    """Ajoute un tirage dans le csv des tirages de l'EuroMillions
    
    Args:
        item (Item) le tirage à ajouter s'il est valide.

    Returns:
        str: Un message pour signaler que le tirage a été ajouté.
    
    """
    if not(validate_date(item.date)):
        raise HTTPException(status_code=422, detail=f"Unprocessable Entity : date format '{item.date}' not recognized. Should be YYYY-MM-DD.")
    
    if not os.path.exists(csv_file):
        f = open(csv_file, "w")
        f.close()
    
    if (validate_drawing(item)):
        raise HTTPException(status_code=422, detail="Unprocessable Entity : the numbers should be different from each others.")

    if (item.e1 == item.e2):
        raise HTTPException(status_code=422, detail="Unprocessable Entity : the star numbers should be different from each others.")
    
    with open(csv_file, "a") as f:
        row = [item.date, item.n1, item.n2, item.n3, item.n4, item.n5,
            item.e1, item.e2, item.winner, item.gain]
        writer = csv.writer(f)
        writer.writerow(row)

    return({"message": "Item added."})


@app.post("/api/model/retrain")
async def regenerate_model():
    """Relance l'entrainement du modèle de prédiction.
    
    Returns:
        str: Un message pour dire que le modèle a fini son entrainement.
    """

    # Regenerate the model
    modele.init_modele()

    return({"message": "Done."})

