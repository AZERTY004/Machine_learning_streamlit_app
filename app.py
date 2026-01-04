import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Détection d'Attaques Réseau", layout="wide")

@st.cache_resource
def load_assets():
    # Chargement avec les noms de fichiers spécifiques à ton projet
    model = joblib.load("model_final.pkl")
    scaler = joblib.load("scaler.pkl")
    ohencoder = joblib.load("ohencode.pkl")
    labelencoder = joblib.load("labelencode.pkl")
    feature_order = joblib.load("feature_order1.pkl")
    
    # Récupération des colonnes attendues par le scaler
    if hasattr(scaler, "feature_names_in_"):
        expected_features = list(scaler.feature_names_in_)
    else:
        expected_features = feature_order
        
    # EXTRACTION DES MOYENNES (La mémoire du dataset d'entraînement)
    if hasattr(scaler, "mean_"):
        # On crée un dictionnaire {nom_colonne: valeur_moyenne}
        means_dict = dict(zip(expected_features, scaler.mean_))
    else:
        means_dict = {col: 0.0 for col in expected_features}
        
    return model, scaler, ohencoder, labelencoder, expected_features, means_dict

model, scaler, ohencoder, labelencoder, expected_features, means_dict = load_assets()

# Définir les catégories
CAT_COLS = ['proto', 'service']

# Identifier les colonnes numériques à afficher dans le formulaire
NUM_COLS_TO_INPUT = [c for c in expected_features if not c.startswith(('proto_', 'service_'))]

# --- 2. INTERFACE ---
st.title("🛡️ Prédiction des attaques réseau")

st.markdown("****Détection des attaques réseau à l’aide de l’apprentissage automatique, et plus précisément du modèle Random Forest.****")
st.markdown("---")

tab1, tab2 = st.tabs(["Saisie Manuelle", "Charger CSV"])

# --- TAB 1 : SAISIE MANUELLE ---
with tab1:
    st.subheader("« Faites confiance au machine learning, là où l’intuition humaine atteint ses limites. »")
    
    # --- BLOC PRÉCISION COMPACT ET ÉLÉGANT ---
    st.markdown("""
        <div style="
            display: inline-block;
            padding: 8px 20px;
            border-radius: 50px;
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            font-family: sans-serif;
            font-size: 14px;
            font-weight: bold;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 15px;
            border: 1px solid rgba(255,255,255,0.2);
        ">
            🎯 Précision du modèle Random Forest : 99.8%
        </div>
        """, unsafe_allow_html=True)

    st.info("💡 Les champs sont pré-remplis avec des valeurs moyennes (du dataset RT-IoT2022.csv) pour faciliter vos tests.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        proto = st.selectbox("Protocole", ['tcp', 'udp', 'icmp'], key="manual_proto")
        service = st.selectbox("Service", ['mqtt', '-', 'http', 'dns', 'ntp', 'ssl', 'dhcp', 'irc', 'ssh', 'radius'], key="manual_service")
    
    manual_data = {'proto': proto, 'service': service}
    
    with col2:
        for col in NUM_COLS_TO_INPUT:
            # On récupère la moyenne, on la convertit en float pour Streamlit
            default_mean = float(means_dict.get(col, 0.0))
            
            manual_data[col] = st.number_input(
                f"{col}", 
                value=default_mean, 
                format="%.6f", 
                key=f"input_{col}"
            )
    
    input_df = pd.DataFrame([manual_data])

# --- TAB 2 : CHARGER CSV / EXCEL ---
with tab2:
    st.subheader("Analyse de fichiers groupés")
    # MODIFICATION ICI : Ajout de xlsx dans les types autorisés
    uploaded_file = st.file_uploader("Téléchargez votre fichier (CSV ou Excel)", type=["csv", "xlsx"])
    
    if uploaded_file:
        # MODIFICATION ICI : Logique pour lire soit CSV soit Excel
        if uploaded_file.name.endswith('.csv'):
            input_df = pd.read_csv(uploaded_file)
        else:
            input_df = pd.read_excel(uploaded_file)
            
        st.write("Aperçu des données :", input_df.head(3))

# --- 3. TRAITEMENT ET PRÉDICTION ---
st.markdown("---")
if st.button("🔍 Lancer la prédiction", type="primary", use_container_width=True):
    try:
        # Étape A : Encodage One-Hot
        ohe_features = ohencoder.transform(input_df[CAT_COLS])
        ohe_df = pd.DataFrame(ohe_features, columns=ohencoder.get_feature_names_out(CAT_COLS))
        
        # Étape B : Extraction des colonnes numériques
        df_num = input_df[ [c for c in NUM_COLS_TO_INPUT if c in input_df.columns] ].copy()
        df_num = df_num.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Étape C : Fusion et alignement strict
        df_final = pd.concat([df_num.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)
        df_final = df_final.reindex(columns=expected_features, fill_value=0)
        
        # Étape D : Scaling et Conversion NumPy
        # On force en float64 pour éviter l'erreur isnan
        X = df_final.values.astype(np.float64)
        X_scaled = scaler.transform(X)
        
        # Étape E : Prédiction
        preds = model.predict(X_scaled)
        labels = labelencoder.inverse_transform(preds)
        
        # --- 4. AFFICHAGE DES RÉSULTATS ---
        st.subheader("Résultats de l'analyse")
        
        if len(labels) > 1:
            # Cas du CSV
            input_df['Prediction'] = labels
            st.success(f"Analyse de {len(labels)} lignes terminée.")
            st.dataframe(input_df)
        else:
            # Cas du Formulaire
            res = labels[0]
            if res.lower() == 'normal':
                st.success(f"✅ Résultat : **{res}** (Trafic légitime)")
            else:
                st.error(f"⚠️ Alerte Sécurité : **{res}** détecté")
                
    except Exception as e:
        st.error(f"Erreur de traitement : {e}")