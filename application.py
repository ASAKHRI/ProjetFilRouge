

import streamlit as st
import pandas as pd
<<<<<<< HEAD
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pickle
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
# from lightgbm import LGBMClassifier => pickle corrompu a refaire
import joblib
import xgboost as xgb




with open('models/xgb_v3.pkl', 'rb') as file:
        model = pickle.load(file)

col1, col2 = st.columns(2)        

with col1:

    st.subheader('Âge')
    selected_age = st.slider('Age_client', min_value=18, max_value=100)

    st.subheader("Nombre d'enfants")
    nb_enfants = st.number_input("Nombre d'enfants", min_value=0, max_value=10, value=0, step=1)
    st.write("Nombre d'enfants", nb_enfants)


    st.subheader("Membre de la famille")
    famille = st.number_input("Saisissez une valeur numérique", min_value=1, max_value=12, value=1, step=1)
    st.write("Membre de la famille", famille)
    # Sidebar pour la taille du ménage
    st.subheader('Ancienneté proessionnelle')
    selected_days_employed = st.slider('Actif depuis', min_value=0, max_value=60, )

with col2 :    

    st.subheader("Montant du bien voulu")
    montant_bien = st.number_input("Montant du bien", min_value=0.0, max_value=100000000.0, value=50.0, step=1.0)
    st.write("Montant du bien voulut", montant_bien)

    # Catégorie name_contract_type
    st.subheader('Type de contrat')
    type_contrat_options = ['Cash Loans', 'Revolving Loans']
    selected_contract = st.selectbox('Sélectionnez le pays:', type_contrat_options)



########## numériques
 # Sidebar pour la taille du ménage
 
    st.subheader('cotation_1')
    selected_cotation_1 = st.slider('cotation_1:', min_value=0.0, max_value=1.0, step = 0.001)
    
    st.subheader('cotation_2')
    selected_cotation_2 = st.slider('cotation_2:', min_value=0.0, max_value=1.0, step = 0.001)

     # Sidebar pour la taille du ménage
    st.subheader('cotation_3')
    selected_cotation_3 = st.slider('cotation_3', min_value=0.0, max_value=1.0, step = 0.001)
    
         # Sidebar pour l'âge

    # Catégorie Flag own car
    st.subheader('Véhiculé')
    véhiculé_options = ['Oui','Non']
    selected_Véhiculé = st.selectbox('Etes-vous le propriétaire de votre véhicule', véhiculé_options)


  



    st.subheader('Valeur Patrimoine')
    # Utilisation du widget st.number_input()
    valeur_patrimoine = st.number_input("Valeur partimoine", min_value=0.0, max_value=100000000.0, value=1000.0, step=1.0)
    # Affichage de la valeur saisie
    st.write("Montant_credit", valeur_patrimoine)


    st.subheader('Montant_credit')
    # Utilisation du widget st.number_input()
    montant_credit_demandé = st.number_input("Montant Crédit", min_value=0.0, max_value=100000000.0, value=1000.0, step=1.0)
    # Affichage de la valeur saisie
    st.write("Montant_credit", montant_credit_demandé)

=======
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
>>>>>>> 9a918062dfc4ef2f9ea420636d3311871a84bdf2

    st.subheader('periode de credit')
    # periode de credit
    periode_credit = st.number_input("periode de credit", min_value=1, max_value=60, step=1)
    # Affichage de la valeur saisie
    st.write("période de credit", periode_credit)

    st.subheader('annuité')
    # Calcul du montant de l'annuité
<<<<<<< HEAD
    montant_annuité = montant_credit_demandé / periode_credit
    # Affichage du montant de l'annuité
    st.write("Montant de l'annuité", montant_annuité)


# traitement des donées numériques

# Définir les données numériques à mettre à l'échelle
data_num_RobScaler = [[nb_enfants],[famille],[selected_age], [selected_days_employed], [montant_bien], [montant_credit_demandé], [periode_credit],[valeur_patrimoine]]
data_num_MinMax = [[selected_cotation_1], [selected_cotation_2], [selected_cotation_3]]
# Créer un scaler robuste
RobustScaler = RobustScaler()
# Mettre à l'échelle les données
data_numerique_Robscaled = RobustScaler.fit_transform(data_num_RobScaler)
# créer un scaler min max
MinMaxScaler = MinMaxScaler()
# Mettre à l'échelle les données
data_numerique_Robscaled = RobustScaler.fit_transform(data_num_MinMax)

# Créer un DataFrame pour les données numériques normalisées
data_numerique_normalise_df = pd.DataFrame({
    'CNT_CHILDREN': [nb_enfants],
    'AMT_INCOME_TOTAL': [valeur_patrimoine],
    'AMT_CREDIT_x': [montant_credit_demandé],
    'AMT_ANNUITY_x':[montant_annuité],
    'AMT_GOODS_PRICE': [montant_bien],
    'DAYS_BIRTH': [selected_age],
    'DAYS_EMPLOYED': [selected_days_employed],
    'CNT_FAM_MEMBERS' : [famille],    
    'EXT_SOURCE_1': [selected_cotation_1],
    'EXT_SOURCE_2': [selected_cotation_2],
    'EXT_SOURCE_3': [selected_cotation_3],   
})

# Créer un DataFrame pour les données catégorielles
data_catégorielles_df = pd.DataFrame({
    'NAME_CONTRACT_TYPE': [selected_contract],
    'FLAG_OWN_CAR': [selected_Véhiculé]
})

label_encoder = LabelEncoder()
data_catégorielles_encoded_df = data_catégorielles_df.apply(label_encoder.fit_transform)

# Combinez les DataFrames numériques et catégoriels
data_df = pd.concat([data_catégorielles_encoded_df,data_numerique_normalise_df], axis=1)


# Bouton pour lancer la prédiction
if st.button('Prédire'):
    # Faites des prédictions avec votre modèle
    prediction = model.predict_proba(data_df)
    # Afficher le résultat de la prédiction
    st.write('Résultat de la prédiction :', prediction)
=======
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
>>>>>>> 9a918062dfc4ef2f9ea420636d3311871a84bdf2

