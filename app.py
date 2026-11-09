import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Devoir ML - Prédictions", layout="centered")

# Dossier contenant les 4 sous-dossiers du devoir (à placer à côté de app.py)
DOSSIER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modeles")

# ---------------------------------------------------------------------------
# Description des 4 modeles
#   - "variables" : dans l'ORDRE attendu par le modèle
#        ("nom", "cat", indice_encodeur)                  -> variable catégorielle
#        ("nom", "num", min, max, valeur_par_défaut, pas) -> variable numérique
#   - "type"      : regression ou classification
#   - "enc_cible" : (classification) indice de l'encodeur qui décode la classe prédite
# ---------------------------------------------------------------------------
MODELES = {
    "Prix d'un trajet de taxi": {
        "dossier": "taxi_price", "suffixe": "taxi_price", "modele": "rf_model_taxi_price.joblib",
        "type": "regression", "cible": "Trip_Price", "unite": "$",
        "variables": [
            ("Trip_Distance_km", "num", 0.0, None, 25.0, 0.5),
            ("Time_of_Day", "cat", 0),
            ("Day_of_Week", "cat", 1),
            ("Passenger_Count", "num", 1, 4, 2, 1),
            ("Traffic_Conditions", "cat", 2),
            ("Weather", "cat", 3),
            ("Base_Fare", "num", 0.0, None, 3.5, 0.1),
            ("Per_Km_Rate", "num", 0.0, None, 1.2, 0.05),
            ("Per_Minute_Rate", "num", 0.0, None, 0.3, 0.01),
            ("Trip_Duration_Minutes", "num", 0.0, None, 60.0, 1.0),
        ],
    },
    "Prix d'une voiture (car_dataset)": {
        "dossier": "car_dataset", "suffixe": "car_dataset", "modele": "rf_model_car_dataset.joblib",
        "type": "regression", "cible": "Selling_Price", "unite": "",
        "variables": [
            ("Kms_Driven", "num", 0, None, 70000, 1000),
            ("Present_Price", "num", 0, None, 9000000, 100000),
            ("Fuel_Type", "cat", 0),
            ("Seller_Type", "cat", 1),
            ("Transmission", "cat", 2),
            ("Age", "num", 0, None, 8, 1),
        ],
    },
    "Ett d'une voiture (car_data)": {
        "dossier": "car_data", "suffixe": "car_data", "modele": "dt_model_car_data.joblib",
        "type": "classification", "enc_cible": 2, "cible": "Etat",
        "variables": [
            ("Marque", "cat", 0),
            ("Annee", "num", 2009, 2020, 2015, 1),
            ("Boite", "cat", 1),
            ("Prix", "num", 1, 24000000, 5000000, 100000),
            ("Adresse", "cat", 3),
        ],
    },
    "Etat d'un téléphone": {
        "dossier": "telephone", "suffixe": "telephone", "modele": "rf_model_telephone.joblib",
        "type": "classification", "enc_cible": 2, "cible": "Etat",
        "variables": [
            ("Prix", "num", 1, 795000, 150000, 5000),
            ("Adresse", "cat", 0),
            ("Marque", "cat", 1),
            # Noms réels inconnus : à renommer selon votre jeu de données
            ("Variable_4", "num", 5.0, 8.0, 6.5, 0.1),
            ("Variable_5", "num", 7.0, 16.0, 10.0, 0.1),
            ("Variable_6", "num", 1.0, 348.0, 100.0, 1.0),
        ],
    },
}


# Charger les objets sauvegardés du modèle choisi (une seule fois grâce au cache)
@st.cache_resource
def load_objects(nom):
    cfg = MODELES[nom]
    chemin = os.path.join(DOSSIER, cfg["dossier"])
    # importer les encodeurs
    encoders = joblib.load(os.path.join(chemin, f"encoders_{cfg['suffixe']}.joblib"))
    # importer le scaler
    scaler = joblib.load(os.path.join(chemin, f"scaler_{cfg['suffixe']}.joblib"))
    # importer le modèle
    model = joblib.load(os.path.join(chemin, cfg["modele"]))
    # importer les valeurs uniques
    uniques = joblib.load(os.path.join(chemin, f"uniques_{cfg['suffixe']}.joblib"))
    return encoders, scaler, model, uniques


# Fonction de prédiction (une seule ligne)
def Pred_func(nom, valeurs):
    cfg = MODELES[nom]
    encoders, scaler, model, uniques = load_objects(nom)
    # Encoder les variables catégorielles
    x = []
    for valeur, var in zip(valeurs, cfg["variables"]):
        if var[1] == "cat":
            valeur = encoders[var[2]].transform([valeur])[0]
        x.append(valeur)
    # vecteur des valeurs numériques (2D array)
    x_new = np.array(x, dtype=float).reshape(1, -1)
    # Mettre à l'échelle (les modèles ont été entraînés sur des données normalisées)
    x_new = scaler.transform(x_new)
    # Prédire
    y_pred = model.predict(x_new)[0]
    if cfg["type"] == "classification":
        # Retrouver le nom de la classe (ex : "Neuf", "D'occasion", ...)
        return encoders[cfg["enc_cible"]].inverse_transform([int(y_pred)])[0]
    # Arrondir
    return round(float(y_pred), 2)


# Fonction de prédiction multiple
def Pred_func_csv(nom, df):
    n = len(MODELES[nom]["variables"])
    predictions = []
    # Boucle sur les lignes du dataframe
    for row in df.iloc[:, :n].values:
        predictions.append(Pred_func(nom, list(row)))
    df[MODELES[nom]["cible"]] = predictions
    return df


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
st.title("🤖 Devoir ML - Prédictions")

# Choix du modèle dans la barre latérale
nom = st.sidebar.selectbox("Choisir le modèle", list(MODELES.keys()))
cfg = MODELES[nom]
encoders, scaler, model, uniques = load_objects(nom)
noms_variables = [v[0] for v in cfg["variables"]]

st.header(nom)

# Créer les onglets
tab1, tab2 = st.tabs(['Simple Prédiction', 'Prédiction multiple'])

# ---------------- Onglet 1 : prédiction simple ----------------
with tab1:
    st.subheader("Prédire avec une seule entrée")
    st.write(f"Variables du modèle : {', '.join(noms_variables)}")

    with st.form(f'form_simple_{nom}'):
        cols = st.columns(2)
        valeurs = []
        for i, var in enumerate(cfg["variables"]):
            with cols[i % 2]:
                if var[1] == "cat":
                    valeurs.append(st.selectbox(var[0], options=uniques[var[2]], key=f"{nom}_{var[0]}"))
                else:
                    valeurs.append(st.number_input(var[0], min_value=var[2], max_value=var[3],
                                                   value=var[4], step=var[5], key=f"{nom}_{var[0]}"))
        submit = st.form_submit_button('Predict')

    if submit:
        y_pred = Pred_func(nom, valeurs)
        st.success(f"{cfg['cible']} : {y_pred} {cfg.get('unite', '')}")

# ---------------- Onglet 2 : prédiction multiple ----------------
with tab2:
    st.subheader("Prédire avec plusieurs entrées (fichier CSV)")
    st.write(f"Le CSV doit contenir ces colonnes, dans cet ordre : {', '.join(noms_variables)}")

    file = st.file_uploader('Upload a csv file', type=['csv'], key=f"file_{nom}")

    if file is not None:
        # Lire le fichier csv
        df = pd.read_csv(file)
        st.write("Aperçu des données chargées :")
        st.dataframe(df.head())

        if st.button('Predict', key=f"btn_{nom}"):
            if df.shape[1] < len(noms_variables):
                st.error(f"Il faut au moins {len(noms_variables)} colonnes.")
            else:
                try:
                    df_pred = Pred_func_csv(nom, df)
                    st.write("Résultats des prédictions :")
                    st.dataframe(df_pred)
                    # Télécharger le fichier des prédictions
                    st.download_button(label='Download a csv file',
                                       data=df_pred.to_csv(index=False).encode('utf-8'),
                                       file_name='predictions.csv',
                                       mime='text/csv')
                except ValueError as e:
                    st.error(f"Erreur dans le fichier (valeur inconnue ou mauvaise colonne) : {e}")
