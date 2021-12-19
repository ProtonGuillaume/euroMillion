TP Money Money Money
--------------------

Mathieu LAPASSADE (ICC) et Guillaume PROTON (IA)



Choix techniques de l'application
---------------------------------

Afin de réaliser ce TP, nous avons choisi d'utiliser scikitlearn et plus particulièrement le classifieur SVM (Suppotr Vector Machines), en combinaison avec FastAPI.



Installation de l'application
-----------------------------

Afin de pouvoir faire tourner cette application, il vous faudra installer plusieurs librairies.
Vous pouvez le faire soit sur votre système directement ou dans un environnement virtuel (recommandé).

Les librairies nécessaires sont :
    - FastAPI (pip install fastapi[all])
    - scikitlearn (pip install sklearn)
    - numpy (pip install numpy)
    - pandas (pip install pandas)

Afin de lancer le serveur FastAPI, il vous faudra vous placer dans le dossier contenant le fichier main.py et taper la commande:
    
    uvicorn main:app --reload

Ensuite vous pourrez vous rendre sur la page web de FastAPI en local sur l'adresse par défaut http://127.0.0.1:8000/docs