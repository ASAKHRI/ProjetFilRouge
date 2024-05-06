

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import RobustScaler, MinMaxScaler, LabelEncoder

with open('models/xgb_v3.pkl','rb') as file :
    model = pickle.load(file)

st.set_page_config(
    page_title = " :bank: Prédiction de solvabilité client",
    page_icon= ":bank:",
    layout="wide",
    initial_sidebar_state="expanded")

st.title(":classical_building: Prediction de solvabilité")



col1, col2 = st.columns(2)

with col1:
    st.subheader('Type de contrat')
    NAME_CONTRACT_TYPE = ['Cash Loans', 'Revolving Loans']
    selected_contract = st.selectbox('Sélectionnez le pays:', NAME_CONTRACT_TYPE)

    # Catégorie Flag own car
    st.subheader('Véhiculé')
    FLAG_OWN_CAR = ['Oui','Non']
    selected_Véhicule = st.selectbox('Etes-vous le propriétaire de votre véhicule', FLAG_OWN_CAR)

########## numériques
    st.subheader("nombre d'enfants")
    # Utilisation du widget st.number_input()
    CNT_CHILDREN = st.number_input("Nombre d'enfants", min_value=0, max_value=10,  step=1)
    # Affichage de la valeur saisie
    st.write("Ancienneté professionnelle", CNT_CHILDREN)
    
with col2:    

    st.subheader('Revenus total')
    # Utilisation du widget st.number_input()
    AMT_INCOME_TOTAL = st.number_input("Age client", min_value=0, max_value=1000000000)
    # Affichage de la valeur saisie
    st.write("Revenu total", AMT_INCOME_TOTAL)

    st.subheader('Montant Credit')
    # Utilisation du widget st.number_input()
    AMT_CREDIT_x = st.number_input("Montant du crédit", min_value=1000, max_value=1000000000)
    # Affichage de la valeur saisie
    st.write("Montant du crédit", AMT_CREDIT_x)

    st.subheader('periode de credit')
    # periode de credit
    periode_credit = st.number_input("periode de credit", min_value=1, max_value=60, step=1)
    # Affichage de la valeur saisie
    st.write("période de credit", periode_credit)

    st.subheader('annuité')
    # Calcul du montant de l'annuité
    AMT_ANNUITY_x = AMT_CREDIT_x / periode_credit
    # Affichage du montant de l'annuité
    st.write("Montant de l'annuité", AMT_ANNUITY_x)

    st.subheader('Montant bien voulu')
    # Utilisation du widget st.number_input()
    AMT_GOODS_PRICE = st.number_input("Saisissez une valeur numérique", min_value=1000.0, max_value=100000000.0, step=500.0)
    # Affichage de la valeur saisie
    st.write("Montant du bien voulut", AMT_GOODS_PRICE)

    st.subheader('Age')
    # Utilisation du widget st.number_input()
    DAYS_BIRTH = st.number_input("Age client", min_value=18, max_value=90,  step=1)
    # Affichage de la valeur saisie
    st.write("Age", DAYS_BIRTH)


    st.subheader('Ancienneté proessionnelle')
    # Utilisation du widget st.number_input()
    DAYS_EMPLOYED = st.number_input("Anciennet&é professionnelle", min_value=0, max_value=50,  step=1)
    # Affichage de la valeur saisie
    st.write("Ancienneté professionnelle", DAYS_EMPLOYED)

    st.subheader("Famille")
    # Utilisation du widget st.number_input()
    CNT_FAM_MEMBERS = st.number_input("Membre foyer", min_value=0, max_value=10,  step=1)
    # Affichage de la valeur saisie
    st.write("Nombre de personnes dans le foyer", CNT_FAM_MEMBERS)

#####

    st.subheader("COT1")
    # Utilisation du widget st.number_input()
    EXT_SOURCE_1 = st.number_input("COT1", min_value=0.0, max_value=1.0,  step=0.001)
    # Affichage de la valeur saisie
    st.write("COT1", EXT_SOURCE_1)
     # Sidebar pour la taille du ménage
    

    st.subheader("COT2")
    # Utilisation du widget st.number_input()
    EXT_SOURCE_2 = st.number_input("COT2", min_value=0.0, max_value=1.0,  step=0.001)
    # Affichage de la valeur saisie
    st.write("COT2", EXT_SOURCE_2)

    st.subheader("COT3")
    # Utilisation du widget st.number_input()
    EXT_SOURCE_3 = st.number_input("COT3", min_value=0.0, max_value=1.0,  step=0.001)
    # Affichage de la valeur saisie
    st.write("COT3", EXT_SOURCE_3)
     # Sidebar pour la taille du ménage

################
################
label_encoder = LabelEncoder()

# Encodage de la variable catégorielle 'selected_contract'
encoded_contract_type = label_encoder.fit_transform(pd.Series(selected_contract))

# Encodage de la variable catégorielle 'selected_Véhicule'
encoded_flag_own_car = label_encoder.fit_transform(pd.Series(selected_Véhicule))
# Création de DataFrames pour les variables encodées
encoded_contract_type_df = pd.DataFrame(encoded_contract_type, columns=['NAME_CONTRACT_TYPE'])
encoded_flag_own_car_df = pd.DataFrame(encoded_flag_own_car, columns=['FLAG_OWN_CAR'])

# Création du RobustScaler
robust_scaler = RobustScaler()

# Sélection des variables numériques à normaliser
numerical_features = pd.DataFrame({
    'CNT_CHILDREN': [CNT_CHILDREN],
    'AMT_INCOME_TOTAL': [AMT_INCOME_TOTAL],
    'AMT_CREDIT_x': [AMT_CREDIT_x],
    'AMT_ANNUITY_x': [AMT_ANNUITY_x],
    'AMT_GOODS_PRICE': [AMT_GOODS_PRICE],
    'DAYS_BIRTH': [DAYS_BIRTH],
    'DAYS_EMPLOYED': [DAYS_EMPLOYED],
    'CNT_FAM_MEMBERS': [CNT_FAM_MEMBERS]
})

# Normalisation des variables numériques avec RobustScaler
scaled_numerical_features = robust_scaler.fit_transform(numerical_features)
scaled_numerical_features = pd.DataFrame(scaled_numerical_features, columns=numerical_features.columns)

# Création du MinMaxScaler
minmax_scaler = MinMaxScaler()

# Sélection des variables à normaliser
ext_sources = pd.DataFrame({
    'EXT_SOURCE_1': [EXT_SOURCE_1],
    'EXT_SOURCE_2': [EXT_SOURCE_2],
    'EXT_SOURCE_3': [EXT_SOURCE_3]
})

# Normalisation avec MinMaxScaler
scaled_ext_sources = minmax_scaler.fit_transform(ext_sources)
scaled_ext_sources = pd.DataFrame(scaled_ext_sources, columns=ext_sources.columns)

# Ajouter un bouton de prédiction
if st.button('Prédire'):
    # Concaténer les données encodées et normalisées
    features = pd.concat([encoded_contract_type_df, encoded_flag_own_car_df, scaled_numerical_features, scaled_ext_sources], axis=1)
    # Faire la prédiction avec le modèle
    prediction = model.predict_proba(features)

    # Afficher le résultat de la prédiction
    st.write("La prédiction de solvabilité du client est :", prediction)

