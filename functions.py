from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold 
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split


from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,roc_curve, auc
from sklearn import metrics
from sklearn.model_selection import KFold , cross_val_score, learning_curve, StratifiedKFold







######################################################



def supprimer_colonnes_manquantes(df, seuil):

    pourcentage_manquant = df.isnull().mean()
    colonnes_a_supprimer = pourcentage_manquant[pourcentage_manquant >= seuil].index
    df = df.drop(columns=colonnes_a_supprimer)
    return df


def print_classification_report(target_test, target_predite):

    print(metrics.classification_report(target_test,target_predite))
    print('- - - - - - - - -')
    print('- - - - - - - - -')
    accuracy = accuracy_score(target_test, target_predite)
    #print("Accuracy: %.2f%%" % (accuracy * 100.0), (np.mean(scores),(np.std(scores))))
    print('- - - - - - - - -')
    print('- - - - - - - - -')


def plot_matrix (target_test,target_predite):
    print(confusion_matrix(target_test,target_predite))


######################################################

def plot_learning_curve (nom_model, jeu_entrainement, target_entrainement) :
    # Génération de la courbe d'apprentissage
    train_sizes, train_scores, test_scores = learning_curve(nom_model, jeu_entrainement, target_entrainement, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

    # Calcul des scores moyens et des écarts types pour les ensembles d'entraînement et de test
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Tracé de la courbe d'apprentissage avec redimensionnement de la figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Courbe d'apprentissage")
    ax.set_xlabel("Taille de l'ensemble d'entraînement")
    ax.set_ylabel("Score")
    ax.grid()
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Score d'entraînement")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Score de validation croisée")
    ax.legend(loc="best")
    plt.show()

##########################################################

def plot_features_importance(nom_model, jeu_entrainement) :   
    # Obtenir l'importance des caractéristiques
    importances = abs(nom_model.coef_[0])

    # Trier l'importance des caractéristiques en ordre décroissant
    indices = np.argsort(importances)[::-1]

    # Nom des caractéristiques
    feature_names = list(jeu_entrainement.columns)

    # Tracer le graphique de l'importance des caractéristiques
    plt.figure(figsize=(10, 6))
    plt.title("Importance des caractéristiques")
    plt.bar(range(jeu_entrainement.shape[1]), importances[indices], color="b", align="center")
    plt.xticks(range(jeu_entrainement.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, jeu_entrainement.shape[1]])
    plt.ylim([0, 0.6])
    plt.show()

###########################################################


def plot_features_importance_10(nom_model, jeu_entrainement, top_n=10):
    # Obtenir l'importance des caractéristiques
    importances = abs(nom_model.coef_[0])

    # Trier l'importance des caractéristiques en ordre décroissant
    indices = np.argsort(importances)[::-1][:top_n]

    # Nom des caractéristiques
    feature_names = list(jeu_entrainement.columns)

    # Tracer le graphique de l'importance des caractéristiques
    plt.figure(figsize=(10, 6))
    plt.title("Importance des caractéristiques")
    plt.bar(range(len(indices)), importances[indices], color="b", align="center")
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, len(indices)])
    plt.ylim([0, 0.6])
    plt.show()


##########################################################    


def plot_roc_auc(nom_model,jeu_test,target_test):
    # Prédiction des probabilités de classe sur les données de test


    y_pred_prob = nom_model.predict_proba(jeu_test)[:, 1]

    # Calcul de la courbe ROC et de l'aire sous la courbe (AUC)
    fpr, tpr, thresholds = roc_curve(target_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Tracé de la courbe ROC
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='Courbe ROC (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC')
    plt.legend(loc="lower right")
    plt.show()



##########################################################

def plot_nan_percent(df_nan, title_name, tight_layout = True, figsize = (20,8), grid = False, rotation = 90):
    
    '''
    Function to plot Bar Plots of NaN percentages for each Column with missing values
    
    Inputs:
        df_nan: 
            DataFrame of NaN percentages
        title_name: 
            Name of table to be displayed in title of plot
        tight_layout: bool, default = True
            Whether to keep tight layout or not
        figsize: tuple, default = (20,8)
            Figure size of plot    
        grid: bool, default = False
            Whether to draw gridlines to plot or not
        rotation: int, default = 0
            Degree of rotation for x-tick labels

    '''
    
    #checking if there is any column with NaNs or not.
    if df_nan.percent.sum() != 0:
        print(f"Number of columns having NaN values: {df_nan[df_nan['percent'] != 0].shape[0]} columns")
        
        #plotting the Bar-Plot for NaN percentages (only for columns with Non-Zero percentage of NaN values)
        plt.figure(figsize = figsize, tight_layout = tight_layout)
        sns.barplot(x= 'column', y = 'percent', data = df_nan[df_nan['percent'] > 0])
        plt.xticks(rotation = rotation)
        plt.xlabel('Column Name')
        plt.ylabel('Percentage of NaN values')
        plt.title(f'Percentage of NaN values in {title_name}')
        if grid:
            plt.grid()
        plt.show()
    else:
        print(f"The dataframe {title_name} does not contain any NaN values.")



def plot_matrix (target_test,target_predite):
    print(confusion_matrix(target_test,target_predite))        


############################################

def imputer_donnees(X):
    """
    Impute les valeurs manquantes dans un jeu de données.
    
    Parameters:
    X : DataFrame
        Le jeu de données à imputer.
        
    Returns:
    DataFrame
        Le jeu de données avec les valeurs manquantes imputées.
    """
    # Listes des colonnes catégorielles et numériques
    colonnes_catégorielles = [colonne for colonne in X.columns if X[colonne].dtype == 'object']
    colonnes_numériques = [colonne for colonne in X.columns if X[colonne].dtype in ['int64', 'float64']]

    # Imputation par mode pour les variables catégorielles
    imputeur_mode = SimpleImputer(strategy='most_frequent')
    for colonne in colonnes_catégorielles:
        X[colonne] = imputeur_mode.fit_transform(X[[colonne]])

    # Imputation par médiane pour les variables numériques
    imputeur_median = SimpleImputer(strategy='median')
    for colonne in colonnes_numériques:
        X[colonne] = imputeur_median.fit_transform(X[[colonne]])

    return X    