import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from PIL import Image
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# ici on charge les données
data = pd.read_csv("data/data.csv")
data_clean = pd.read_csv("data/data.csv")

# ici on configure la page principale
st.set_page_config(page_title="Projet Final", layout="wide")

# ici on configure les differents tab de la page
page1, page2, page3, page4, page5 = st.tabs([
    "Accueil",
    "Prétraitement des données",
    "Analyse exploratoire",
    "Visualisation des données",
    "Synthèse & Conclusion" 
])

# -----------------------------------------------------------------------------
# Page 1 : Accueil
# -----------------------------------------------------------------------------
with page1:
    # Titre de la page
    st.title("📊 Projet Coffe Shop ")
    st.subheader("Une brève introduction à notre projet et à notre équipe")
    st.write("Lien du dataset selectionné : https://www.kaggle.com/datasets/ahmedmohamed2003/cafe-sales-dirty-data-for-cleaning-training")
    st.divider()  # Ligne de séparation élégante

    # ici on decris le projet
    st.subheader("🔍 À propos du projet")
    st.write("""
    Notre projet consiste à analyser un dataset de transactions provenant d'un coffee shop.
    Nous allons nettoyer les données, les analyser et les
    visualiser pour en tirer des enseignements pertinents.
    """)   


    def resize_image(image_path, size=(150, 150)):
        image = Image.open(image_path)
        image = image.resize(size) 
        return image

    # ici on presente notre equipe
    st.subheader("👥 Équipe du projet")
    st.write(" ")

    # ici on liste chaque personne de l'équipe avec sa photo
    team = [
        {"nom": "Hiba Rahil", "photo": "photo/Hiba.jpeg"},
        {"nom": "Magdat Djaoid", "photo": "photo/Magdat.jpeg"},
        {"nom": "Manal Lazhar", "photo": "photo/Manal.jpeg"},
        {"nom": "Emma Gilbert", "photo": "photo/Emma.jpeg"},
        {"nom": "Alaïs Ervera", "photo": "photo/Alaïs.jpeg"},
    ]
    cols = st.columns(len(team))  
    # ici on affiche chaque membre de l'équipe avec sa photo
    for col, membre in zip(cols, team):
        with col:
            image = resize_image(membre["photo"])
            st.image(image, width=150)
            st.markdown(f"**{membre['nom']}**")

# -----------------------------------------------------------------------------
# Page 2 : Prétraitement des données
# -----------------------------------------------------------------------------
with page2:
    # Titre de la page
    st.subheader("Prétraitement des données")
    st.write("Voici un aperçu des données utilisées pour le traitement. "
         "Cette étape nous a permis de mieux comprendre la structure du dataset, notamment grâce à la fonction `head()`, "
         "qui affiche les premières valeurs")
    st.write(" ")


    st.write("Premiere valeurs du dataset :")
    st.dataframe(data.head()) # ici on Affiche les premières valeurs du dataset
    st.markdown("---")

    st.subheader("1️⃣ Vérification des valeurs manquantes")
    st.write("Avant de se lancer dans la partie de Analyse des données, il est important de vérifier si notre dataset"
    " contient des valeurs manquantes. Pour cela, nous avons utilisé la fonction `isnull().sum()` pour compter le nombre "
    "de valeurs manquantes par colonne et nous avons obtenus çe resultat :")

    # ici on Calcule le nombre total de valeurs manquantes
    total_missing_values = data.isnull().sum().sum()

    missing_per_column = data.isnull().sum()
    st.dataframe(missing_per_column)
    st.warning(f"On remarque qu'il y a énormément de valeurs manquantes dans certaines colonnes, avec un total de **{total_missing_values}** valeurs manquantes.")



    st.subheader("2️⃣ Imputation des valeurs manquantes")
    st.markdown("""
    Pour la gestion de valeurs manquantes on comptait toute les supprimer cependant dans ce dataset il y a trop de valeurs manquantes, 
    supprimer chaque ligne avec une valeur manquante supprimerais trop de lignes et engendrerait une perte de données non negligeable.  
    Nous allons donc plutôt imputer les champs en utilisant les méthodes suivantes :
    - **Colonnes numériques** : Remplacement par la médiane (`fillna(median)`)
    - **Colonnes catégorielles** : Remplacement par la modalité la plus fréquente (`fillna(mode)`)
    """)
    st.write(" ")

    numerical_cols = data_clean.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = data_clean.select_dtypes(include=['object', 'category']).columns

    # ici on impute les valeurs manquantes
    # les valeurs numériques sont remplacées par la médiane
    for col in numerical_cols:
        median_value = data_clean[col].median()
        data_clean[col].fillna(median_value, inplace=True)

    # les valeurs catégorielles sont remplacées par la valeur la plus fréquente
    for col in categorical_cols:
        mode_value = data_clean[col].mode()[0]

        data_clean[col] = data_clean[col].replace(["ERROR", "UNKNOWN", np.nan], mode_value)

    # Calcul du nombre total de valeurs manquantes après traitement
    total_missing_values = data_clean.isnull().sum().sum()

    # Vérification des valeurs manquantes par colonne
    missing_per_column = data_clean.isnull().sum()

    # Affichage des résultats 
    st.dataframe(missing_per_column)
    st.success(f"A l'aide du tableau, on remarque qu'on a plus aucune valeur manquante, (nombre de valeur manquantes : **{total_missing_values}** )")
    # Sauvegarde des modifications dans le fichier CSV
    data_clean.to_csv("data/data_clean.csv", index=False)
    

    # Encodage avec des chiffres pour 'item' et 'payment method'
    label_encoders = {}  # Dictionnaire pour stocker les encoders

    for col in ['Item', 'Payment Method', 'Location']:
        le = LabelEncoder()
        data_clean[col] = le.fit_transform(data_clean[col])  # l'encodage commence a 0
        label_encoders[col] = le  # Stocker l'encoder si besoin d'inverser plus tard
    # Sauvegarde des modifications dans le fichier CSV
    data_clean.to_csv("data/data_clean.csv", index=False)

    # Affichage sur Streamlit
    st.subheader(" 3️⃣  Encodage des variables catégorielles")
    st.write("Pour faciliter le traitement des données, nous avons encodé les variables catégorielles en utilisant des chiffres. "
             "Voici à quoi ressemble le dataset après encodage :")
    st.write(data_clean.head())

    # Affichage des mappings utilisés
    st.write("Mappings des valeurs encodées :")
    mappings_df = pd.DataFrame({
        "Colonne": [col for col in label_encoders.keys()],
        "Valeurs originales": [list(le.classes_) for le in label_encoders.values()],
        "Valeurs encodées": [list(le.transform(le.classes_)) for le in label_encoders.values()]
    })
    st.dataframe(mappings_df)

    # 🔄 Correction des types de données avant
    st.subheader("4️⃣ Typage de chaque colonne")
    st.write("cette partie consiste à convertir les types de données de chaque colonne en fonction de leur contenu. "
             "Nous avons utilisé la fonction `dtypes` pour afficher les types de données avant et après le nettoyage.")
    st.write(" ")

    # Séparation des colonnes numériques, catégorielles et date
    numerical_cols = ['Quantity', 'Price Per Unit', 'Total Spent', 'Item', 'Payment Method','Location'] 
    categorical_cols = ['Transaction ID']
    date_cols = ['Transaction Date']

    # Conversion des types
    for col in numerical_cols:
        data_clean[col] = pd.to_numeric(data_clean[col], errors='coerce')  # Convertit en (int/float)

    for col in categorical_cols:
        data_clean[col] = data_clean[col].astype('category')  # Convertit en catégorie

    for col in date_cols:
        data_clean[col] = pd.to_datetime(data_clean[col], errors='coerce')  # Convertit en datetime et met NaT si erreur

    # Suppression des lignes où la date est NaT (erreur de conversion ou vide)
    data_clean = data_clean.dropna(subset=date_cols)
    # Sauvegarde des modifications dans le fichier CSV
    data_clean.to_csv("data/data_clean.csv", index=False)

    # Comparaison après nettoyage
    col1, col2 = st.columns(2)
    with col1:
        st.warning("Types avant conversion :")
        st.dataframe(data.dtypes)
    with col2:
        st.success("Types après conversion :")
        st.dataframe(data_clean.dtypes)


# -----------------------------------------------------------------------------
# Page 3 : Analyse exploratoire
# -----------------------------------------------------------------------------

with page3 :
    # Titre de la page
    st.subheader("Analyse exploratoire")
    st.write("statistiques du dataset :")


    info_dict = {
        "Column": data.columns,
        "Non-Null Count": data.count().values,
        "Dtype": [data[col].dtype for col in data.columns]
    }
    data_info = pd.DataFrame(info_dict)
    st.dataframe(data_clean.describe())


    st.subheader("📌 Matrice de Corrélation")
    st.write("La matrice de corrélation ci-dessous nous permet d’analyser la relation entre différentes "
    "variables de notre dataset. Elle nous indique dans quelle mesure deux variables évoluent ensemble.")

    with st.expander("🔎 Options d'analyse", expanded=False):
            selected_features = st.multiselect(
                "Sélectionnez les variables pour la matrice de corrélation :",
                data_clean.select_dtypes(include=['int64', 'float64']).columns.tolist(),
                default=["Item","Quantity", "Price Per Unit", "Total Spent",
                        "Payment Method", "Location"]
            )

   # Vérification que les colonnes sélectionnées existent bien et sont numériques
    selected_features = [col for col in selected_features if col in data_clean.columns and data_clean[col].dtype in ['int64', 'float64']]

    # Calcul de la matrice de corrélation
    correlation_matrix = data_clean[selected_features].corr()

    # Création de la heatmap avec des annotations et une mise en forme corrigée
    fig_corr = ff.create_annotated_heatmap(
        z=correlation_matrix.values[::-1], 
        x=list(correlation_matrix.columns),
        y=list(correlation_matrix.index)[::-1],
        colorscale="RdBu",
        annotation_text=np.round(correlation_matrix.values[::-1], 2),
        showscale=True,
        reversescale=True
    )

    fig_corr.update_layout(
        height=700
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("📊 Interprétation de la Matrice de Corrélation")
    st.write("""
    L'analyse de la matrice de corrélation nous permet de mieux comprendre les relations entre les variables de notre dataset. Voici les principales conclusions que nous pouvons en tirer :
    """)
    
    st.markdown("- ✅ **Plus la quantité achetée est grande, plus la dépense totale est élevée** "
    "→ Corrélation de **0.63** entre `Quantity` et `Total Spent`. Cela signifie qu'en moyenne, acheter "
    "en plus grande quantité entraîne une augmentation du total dépensé.")
    
    st.markdown("- ✅ **Un prix unitaire plus élevé entraîne une dépense totale plus importante**  "
    "→ Corrélation de **0.61** entre `Price Per Unit` et `Total Spent`. Plus un produit est cher à l'unité, plus le total dépensé a tendance à être élevé.")
    
    st.markdown("- 🔍 **Certains items ont tendance à être plus chers que d’autres**  "
    "→ Corrélation de **0.22** entre `Item` et `Price Per Unit`. Bien que cette relation soit plus faible, elle indique que certains articles spécifiques ont des prix unitaires plus élevés.")
    
    st.write("""
    En revanche, nous observons que certaines variables, comme `Payment Method` et `Location`, ont des corrélations très faibles avec les autres paramètres. Cela signifie qu'elles n'ont pas d'impact significatif sur les achats en termes de montants ou de quantités.
    """)

# -----------------------------------------------------------------------------
# Page 4 : Visualisation des données
# -----------------------------------------------------------------------------
with page4 :
    st.subheader("Visualisation des données")

    # Dictionnaire pour remapper les valeurs encodées vers leurs noms réels
    item_mapping = {
        0: "Cake",
        1: "Coffee",
        2: "Cookie",
        3: "Juice",
        4: "Salad",
        5: "Sandwich",
        6: "Smoothie",
        7: "Tea"
    }

    # ici on remplace les valeurs encodées par leur nom réel
    data_clean['Item_Names'] = data_clean['Item'].map(item_mapping)

    # ici on compte le nombre d’achats par item
    items_counts = data_clean['Item_Names'].value_counts().reset_index()
    items_counts.columns = ['Item', 'Nombre d’achats']

    # Création du graphique avec Plotly
    fig_items = px.bar(
        items_counts,
        y="Item",
        x="Nombre d’achats",
        color="Nombre d’achats",
        title="📌 Histogramme des Items les Plus Achetés",
        labels={"Item": "Nom de l’Item", "Nombre d’achats": "Nombre de fois acheté"},
        text_auto=True
    )

    # Personnalisation du layout
    fig_items.update_layout(
        xaxis=dict(tickangle=45),
        yaxis_title="Nombre d’achats",
        xaxis_title="Items",
        margin=dict(l=50, r=50, t=50, b=100),
        height=600
    )

    # Affichage du graphique 
    st.plotly_chart(fig_items, use_container_width=True)


    # pareil que le graph avant mais pour les paiement (mêmes etapes)
    payment_mapping = {
        0: "Cash",
        1: "Credit Card",
        2: "Digital Wallet"
    }
    
    data_clean['Payment Method'] = data_clean['Payment Method'].map(payment_mapping)
    
    payment_counts = data_clean['Payment Method'].value_counts().reset_index()
    payment_counts.columns = ['Méthode de Paiement', 'Nombre de Transactions']

    fig_payment = px.bar(
        payment_counts,
        x="Méthode de Paiement",
        y="Nombre de Transactions",
        color="Nombre de Transactions",
        title="📌 Répartition des Méthodes de Paiement",
        labels={"Méthode de Paiement": "Type de Paiement", "Nombre de Transactions": "Nombre de Transactions"},
        text_auto=True
    )

    fig_payment.update_layout(
        xaxis_title="Méthode de Paiement",
        yaxis_title="Nombre de Transactions",
        margin=dict(l=50, r=50, t=50, b=100),
        height=600
    )

    st.plotly_chart(fig_payment, use_container_width=True)

    # graphique de density plot avec plotly
    fig = px.density_heatmap(data_clean, x="Total Spent", y="Quantity", marginal_x="histogram", marginal_y="histogram",
                             title="📌 Distribution des Dépenses Totales et Quantités")
    fig.update_layout(
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Page 5 : Synthèse & Conclusion
# -----------------------------------------------------------------------------
    with page5:
        st.subheader("Synthese & Conclusion")
        st.write("""
        Travailler en équipe sur ce projet a été une expérience enrichissante, nous permettant d’explorer les différentes
        étapes du traitement des données. Le prétraitement des données nous a particulièrement marqué par sa complexité 
        et le temps qu’il a nécessité. 
        """)
        st.subheader("Enseignements et Perspectives")
        st.write("""
        Grâce à notre analyse, nous avons pu dégager plusieurs tendances intéressantes sur
        les ventes du coffee shop. Nous avons constaté que plus la quantité achetée est élevée, plus le prix total est 
        influencé, ce qui est logique. De même, un prix unitaire plus élevé impacte directement le coût final. Du côté 
        des produits, le jus s’est révélé être l’article le plus acheté, et le mode de paiement le plus utilisé est le 
        Digital Wallet, avec une transaction moyenne avoisinant 3,9 $.
        """)
        st.write(" ")
        st.write("""Enfin, afin de rendre notre travail plus interactif et accessible, nous avons choisi de développer une interface 
        utilisateur plutôt que de simplement rendre un Jupyter Notebook. Nous avons trouvé cette approche plus intuitive 
        et intéressante pour la visualisation et l’interprétation des résultats.""")

        st.success("En fin de compte, nous sommes fiers d’avoir pu étudier ce dataset sur les coffee shops. "
        "Cela nous permettra de continuer à boire du café et de discuter entre nous au lieu de travailler.🤡")


