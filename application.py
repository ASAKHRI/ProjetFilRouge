import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler    





 # Catégorie name_contract_type
    st.subheader('Type de contrat')
    type_contrat_options = ['Cash Loans', 'Revolving Loans']
    selected_contract = st.selectbox('Sélectionnez le pays:', type_contrat_options)

    # Catégorie Flag own car
    st.subheader('Véhiculé')
    véhiculé_options = ['Oui','Non']
    selected_Véhiculé = st.selectbox('Etes-vous le propriétaire de votre véhicule', véhiculé_options)

    # Catégorie Accès au téléphone
    st.subheader('Propriétaire')
    proprétaire_options = ['Oui','Non']
    selected_proptiétaire = st.selectbox('Propriétaire', proprétaire_options)

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
    st.subheader('Âge')
    selected_age = st.slider('Age_client', min_value=18, max_value=100)

         # Sidebar pour la taille du ménage
    st.subheader('Ancienneté proessionnelle')
    selected_household_size = st.slider('Actif depuis', min_value=0, max_value=60, )
    
    # Utilisation du widget st.number_input()
    montant_bien = st.number_input("Saisissez une valeur numérique", min_value=0.0, max_value=100000000.0, value=50.0, step=1.0)
    # Affichage de la valeur saisie
    st.write("Montant du bien voulut", montant_bien)


    st.subheader('Montant_credit')
    # Utilisation du widget st.number_input()
    montant_credit_demandé = st.number_input("Saisissez une valeur numérique", min_value=0.0, max_value=100000000.0, value=1000.0, step=1.0)
    # Affichage de la valeur saisie
    st.write("Montant du bien voulut", montant_credit_demandé)


    st.subheader('periode de credit')
    # periode de credit
    periode_credit = st.number_input("periode de credit", min_value=1, max_value=60, step=1)
    # Affichage de la valeur saisie
    st.write("période de credit", periode_credit)

    st.subheader('annuité')
    # Calcul du montant de l'annuité
    montant_annuité = montant_credit_demandé / periode_credit
    # Affichage du montant de l'annuité
    st.write("Montant de l'annuité", montant_annuité)




from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

# Données catégorielles
type_contrat_options = ['Cash Loans', 'Revolving Loans']
selected_contract = st.selectbox('Sélectionnez le pays:', type_contrat_options)

véhiculé_options = ['Oui','Non']
selected_Véhiculé = st.selectbox('Etes-vous le propriétaire de votre véhicule', véhiculé_options)

proprétaire_options = ['Oui','Non']
selected_proptiétaire = st.selectbox('Propriétaire', proprétaire_options)

# Encodage des données catégorielles
label_encoder = LabelEncoder()

# Pour le codage d'étiquettes
selected_contract_encoded = label_encoder.fit_transform(selected_contract)

# Pour le codage one-hot
onehot_encoder = OneHotEncoder(sparse=False)
selected_Véhiculé_encoded = label_encoder.fit_transform(selected_Véhiculé)
selected_proptiétaire_encoded = label_encoder.fit_transform(selected_proptiétaire)

# Création d'un DataFrame pandas avec les données encodées
data_encoded = pd.DataFrame({
    'Type de contrat': selected_contract_encoded,
    'Véhiculé': selected_Véhiculé_encoded,
    'Propriétaire': selected_proptiétaire_encoded
})

# Affichage des données encodées
st.write("Données catégorielles encodées :", data_encoded)