import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler,OneHotEncoder
from lightgbm import LGBMClassifier



model = joblib.load('pipeline.pkl')
   

st.set_page_config(
    page_title = " :bank: Prédiction de solvabilité client",
    page_icon= ":bank:",
    layout="wide",
    initial_sidebar_state="expanded")



def predict():
    row = np.array([NAME_CONTRACT_TYPE,
    FLAG_OWN_CAR,CNT_CHILDREN,AMT_INCOME_TOTAL,
    AMT_CREDIT_x,AMT_ANNUITY_x,AMT_GOODS_PRICE,
    DAYS_BIRTH,DAYS_EMPLOYED,CNT_FAM_MEMBERS,
    EXT_SOURCE_1,EXT_SOURCE_2,EXT_SOURCE_3
    ])
    # data = {
    #     'NAME_CONTRACT_TYPE': [selected_contract],
    #     'FLAG_OWN_CAR': [selected_Véhicule],
    #     'CNT_CHILDREN': [CNT_CHILDREN],
    #     'AMT_INCOME_TOTAL': [AMT_INCOME_TOTAL],
    #     'AMT_CREDIT_x': [AMT_CREDIT_x],
    #     'AMT_ANNUITY_x': [AMT_ANNUITY_x],
    #     'AMT_GOODS_PRICE': [AMT_GOODS_PRICE],
    #     'DAYS_BIRTH': [DAYS_BIRTH],
    #     'DAYS_EMPLOYED': [DAYS_EMPLOYED],
    #     'CNT_FAM_MEMBERS': [CNT_FAM_MEMBERS],
    #     'EXT_SOURCE_1': [EXT_SOURCE_1],
    #     'EXT_SOURCE_2': [EXT_SOURCE_2],
    #     'EXT_SOURCE_3': [EXT_SOURCE_3]
    # }
    X = pd.DataFrame([row])
    prediction = model.predict(X)[:,1]
 
    if prediction[0] <= 0.5 :
        st.success('credit accordé')
    else :
        st.error('credit refusé')    

st.title(":classical_building: Prédiction de solvabilité")

with st.form(key='my_form'):
    st.subheader('Type de contrat')
    NAME_CONTRACT_TYPE = st.selectbox('Sélectionnez le type de contrat:', ['Cash Loans', 'Revolving Loans'])

    st.subheader('Véhiculé')
    FLAG_OWN_CAR = st.selectbox('Êtes-vous le propriétaire de votre véhicule ?', ['Oui', 'Non'])

    st.subheader("nombre d'enfants")
    CNT_CHILDREN = st.number_input("Nombre d'enfants", min_value=0, max_value=10,  step=1)
    st.write("Ancienneté professionnelle", CNT_CHILDREN)
    
    st.subheader('Revenus total')
    AMT_INCOME_TOTAL = st.number_input("Age client", min_value=0, max_value=1000000000)
    st.write("Revenu total", AMT_INCOME_TOTAL)

    st.subheader('Montant Credit')
    AMT_CREDIT_x = st.number_input("Montant du crédit", min_value=1000, max_value=1000000000)
    st.write("Montant du crédit", AMT_CREDIT_x)

    st.subheader('periode de credit')
    periode_credit = st.number_input("periode de credit", min_value=1, max_value=60, step=1)
    st.write("période de credit", periode_credit)

    st.subheader('annuité')
    AMT_ANNUITY_x = AMT_CREDIT_x / periode_credit
    st.write("Montant de l'annuité", AMT_ANNUITY_x)

    st.subheader('Montant bien voulu')
    AMT_GOODS_PRICE = st.number_input("Saisissez une valeur numérique", min_value=1000.0, max_value=100000000.0, step=500.0)
    st.write("Montant du bien voulut", AMT_GOODS_PRICE)

    st.subheader('Age')
    DAYS_BIRTH = st.number_input("Age client", min_value=18, max_value=90,  step=1)
    st.write("Age", DAYS_BIRTH)


    st.subheader('Ancienneté proessionnelle')
    DAYS_EMPLOYED = st.number_input("Anciennet&é professionnelle", min_value=0, max_value=50,  step=1)
    st.write("Ancienneté professionnelle", DAYS_EMPLOYED)

    st.subheader("Famille")
    CNT_FAM_MEMBERS = st.number_input("Membre foyer", min_value=0, max_value=10,  step=1)
    st.write("Nombre de personnes dans le foyer", CNT_FAM_MEMBERS)

    st.subheader("COT1")
    EXT_SOURCE_1 = st.number_input("COT1", min_value=0.0, max_value=1.0,  step=0.001)
    st.write("COT1", EXT_SOURCE_1)
    

    st.subheader("COT2")
    EXT_SOURCE_2 = st.number_input("COT2", min_value=0.0, max_value=1.0,  step=0.001)
    st.write("COT2", EXT_SOURCE_2)

    st.subheader("COT3")
    EXT_SOURCE_3 = st.number_input("COT3", min_value=0.0, max_value=1.0,  step=0.001)
    st.write("COT3", EXT_SOURCE_3)
    # Autres entrées numériques...
    
    submit_button = st.form_submit_button(label='Prédire')

# Faire la prédiction lorsque le bouton est cliqué
if submit_button:
    predict()       













