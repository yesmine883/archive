import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from traitement_module import *
import joblib
from sklearn.preprocessing import StandardScaler

# Configuration de la page
st.set_page_config(
    page_title="Système de Prédiction - Machine Learning",
    page_icon="🤖",
    layout="wide"
)

# Titre de l'application
st.title("🤖 Système Complet de Machine Learning")
st.markdown("Application de prédiction utilisant tous les modèles de votre module")

# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choisir une section",
    ["Accueil", "Chargement des Données", "Préprocessing", "Visualisation", 
     "Modélisation", "Comparaison des Modèles", "Analyse des Performances", "Prédiction"]
)

# Page Accueil
if page == "Accueil":
    st.header("Bienvenue dans l'application de Machine Learning")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **📊 Données**
        - Chargement CSV
        - Exploration
        - Nettoyage
        """)
    
    with col2:
        st.info("""
        **🔧 Préprocessing**
        - Encodage
        - Normalisation
        - Split Train/Test
        """)
    
    with col3:
        st.info("""
        **🤖 Modèles**
        - 12 algorithmes
        - GridSearch
        - Comparaison
        """)
    
    st.markdown("---")
    st.subheader("Modèles disponibles dans votre module :")
    
    models_list = [
        "Régression Linéaire", "Ridge/Lasso/ElasticNet", "Régression Polynomiale",
        "Arbre de Décision", "Random Forest", "Gradient Boosting", 
        "XGBoost", "LightGBM", "CatBoost", "SVR", "KNN"
    ]
    
    for i, model in enumerate(models_list):
        st.write(f"✅ {model}")

# Page Chargement des Données
elif page == "Chargement des Données":
    st.header("📁 Chargement des Données")
    
    uploaded_file = st.file_uploader("Choisir un fichier CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Charger les données
            df = charger_donnees(uploaded_file)
            st.session_state['df'] = df
            st.session_state['df_original'] = df.copy()  # Sauvegarder une copie originale
            st.success("✅ Données chargées avec succès !")
            
            # Métriques
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Lignes", df.shape[0])
            with col2:
                st.metric("Colonnes", df.shape[1])
            with col3:
                st.metric("Valeurs manquantes", df.isnull().sum().sum())
            with col4:
                st.metric("Doublons", df.duplicated().sum())
            
            # Aperçu des données
            st.subheader("Aperçu des données")
            st.dataframe(df.head())
            
            # Informations sur les types
            st.subheader("Types de données")
            types_df = pd.DataFrame({
                'Colonne': df.columns,
                'Type': df.dtypes,
                'Valeurs uniques': [df[col].nunique() for col in df.columns],
                'Valeurs manquantes': df.isnull().sum().values
            })
            st.dataframe(types_df)
            
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement : {e}")

# Page Préprocessing
elif page == "Préprocessing":
    st.header("🔧 Préprocessing des Données")
    
    if 'df' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord charger des données dans la section 'Chargement des Données'")
    else:
        df = st.session_state['df']
        
        # Encodage des colonnes catégorielles
        st.subheader("Encodage des variables catégorielles")
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            selected_categorical = st.multiselect(
                "Sélectionner les colonnes à encoder :",
                categorical_cols,
                default=categorical_cols
            )
            
            if st.button("Encoder les colonnes sélectionnées"):
                df_encoded = encoder_colonnes(df.copy(), selected_categorical)
                st.session_state['df'] = df_encoded
                st.session_state['encoded_columns'] = selected_categorical
                st.success(f"✅ {len(selected_categorical)} colonne(s) encodée(s) avec succès !")
                
                # Afficher le mapping pour une colonne
                if selected_categorical:
                    sample_col = selected_categorical[0]
                    st.write(f"Mapping pour '{sample_col}':")
                    original_values = st.session_state['df_original'][sample_col].unique()
                    encoded_values = df_encoded[sample_col].unique()
                    mapping_df = pd.DataFrame({
                        'Valeur originale': original_values,
                        'Valeur encodée': encoded_values
                    })
                    st.dataframe(mapping_df)
        else:
            st.info("ℹ️ Aucune colonne catégorielle détectée.")
        
        # Séparation des features et target
        st.subheader("Séparation Features/Target")
        all_columns = st.session_state['df'].columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox("Colonne cible (Y) :", all_columns, index=len(all_columns)-1)
        with col2:
            feature_cols = st.multiselect("Colonnes features (X) :", all_columns, 
                                        default=[col for col in all_columns if col != target_col])
        
        if st.button("Préparer les données pour la modélisation"):
            try:
                X = st.session_state['df'][feature_cols]
                Y = st.session_state['df'][target_col]
                
                # Sauvegarder les noms des features pour la prédiction
                st.session_state['feature_names'] = feature_cols
                st.session_state['target_name'] = target_col
                st.session_state['X_original'] = X.copy()
                st.session_state['Y_original'] = Y.copy()
                
                # Split et scale
                X_train, X_test, Y_train, Y_test = split_scale(X, Y)
                
                # Sauvegarder dans session state
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['Y_train'] = Y_train
                st.session_state['Y_test'] = Y_test
                st.session_state['scaler'] = StandardScaler().fit(X)  # Sauvegarder le scaler
                
                st.success("✅ Données préparées avec succès !")
                
                # Afficher les dimensions
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("X_train", f"{X_train.shape}")
                with col2:
                    st.metric("X_test", f"{X_test.shape}")
                with col3:
                    st.metric("Y_train", f"{Y_train.shape}")
                with col4:
                    st.metric("Y_test", f"{Y_test.shape}")
                    
                st.info(f"🔧 {len(feature_cols)} features sélectionnées pour l'entraînement")
                    
            except Exception as e:
                st.error(f"❌ Erreur : {e}")

# Page Visualisation
elif page == "Visualisation":
    st.header("📊 Visualisation des Données")
    
    if 'df' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord charger des données")
    else:
        df = st.session_state['df']
        
        # Sélection du type de visualisation
        viz_type = st.selectbox("Type de visualisation :", 
                               ["Distribution", "Corrélation", "Boxplot", "Relation"])
        
        if viz_type == "Distribution":
            selected_col = st.selectbox("Sélectionner une colonne :", 
                                      df.select_dtypes(include=[np.number]).columns)
            fig, ax = plt.subplots(figsize=(10, 6))
            df[selected_col].hist(bins=30, ax=ax)
            ax.set_title(f'Distribution de {selected_col}')
            st.pyplot(fig)
            
        elif viz_type == "Corrélation":
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
                st.pyplot(fig)
            else:
                st.warning("Pas assez de colonnes numériques pour la corrélation")
                
        elif viz_type == "Boxplot":
            selected_col = st.selectbox("Sélectionner une colonne :", 
                                      df.select_dtypes(include=[np.number]).columns)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df, y=selected_col, ax=ax)
            st.pyplot(fig)
            
        elif viz_type == "Relation":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Variable X :", numeric_cols)
            with col2:
                y_col = st.selectbox("Variable Y :", numeric_cols)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df, x=x_col, y=y_col, alpha=0.6, ax=ax)
            st.pyplot(fig)

# Page Modélisation
# Page Modélisation - VERSION CORRIGÉE AVEC CLÉS UNIQUES
elif page == "Modélisation":
    st.header("🤖 Entraînement des Modèles")
    
    if 'X_train' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord préparer les données dans la section 'Préprocessing'")
    else:
        # Récupérer les données
        X_train = st.session_state['X_train']
        X_test = st.session_state['X_test']
        Y_train = st.session_state['Y_train']
        Y_test = st.session_state['Y_test']
        feature_names = st.session_state['feature_names']
        
        st.info(f"🎯 Entraînement avec {len(feature_names)} features: {', '.join(feature_names)}")
        
        # Sélection du modèle
        st.subheader("Sélection du modèle")
        model_choice = st.selectbox("Choisir un modèle :", [
            "Régression Linéaire", "Ridge/Lasso/ElasticNet", "Régression Polynomiale",
            "Arbre de Décision", "Random Forest", "Gradient Boosting", 
            "XGBoost", "LightGBM", "CatBoost", "SVR", "KNN"
        ], key="model_choice_select")
        
        # Initialiser les paramètres dans session_state s'ils n'existent pas
        if 'model_params' not in st.session_state:
            st.session_state['model_params'] = {
                'poly_degree': 2,
                'knn_neighbors': 5,
                'rf_estimators': 100
            }
        
        # Variables pour stocker tous les modèles
        all_models = {}
        
        # Afficher les paramètres AVANT le bouton d'entraînement
        st.subheader("🔧 Paramètres du modèle")
        
        # Afficher les paramètres selon le modèle choisi
        if model_choice == "Régression Polynomiale":
            st.session_state['model_params']['poly_degree'] = st.slider(
                "Degré polynomial", 
                2, 5, 
                st.session_state['model_params']['poly_degree'],
                key="poly_degree_slider_unique"
            )
            st.info(f"🎯 Degré sélectionné : {st.session_state['model_params']['poly_degree']}")
            
        elif model_choice == "KNN":
            st.session_state['model_params']['knn_neighbors'] = st.slider(
                "Nombre de voisins", 
                3, 15, 
                st.session_state['model_params']['knn_neighbors'],
                step=2, 
                key="knn_neighbors_slider_unique"
            )
            st.info(f"🎯 Nombre de voisins : {st.session_state['model_params']['knn_neighbors']}")
            
        elif model_choice == "Random Forest":
            st.session_state['model_params']['rf_estimators'] = st.slider(
                "Nombre d'arbres", 
                50, 200, 
                st.session_state['model_params']['rf_estimators'],
                key="rf_estimators_slider_unique"
            )
            st.info(f"🎯 Nombre d'arbres : {st.session_state['model_params']['rf_estimators']}")
        
        # Bouton d'entraînement
        st.markdown("---")
        if st.button("🚀 Entraîner le modèle", type="primary"):
            try:
                with st.spinner("Entraînement en cours..."):
                    
                    if model_choice == "Régression Linéaire":
                        model, Y_pred, rmse = modele_linear_regression(X_train, X_test, Y_train, Y_test)
                        st.success(f"✅ Régression Linéaire - RMSE: {rmse:.4f}")
                        all_models["linear_regression"] = model
                        
                    elif model_choice == "Ridge/Lasso/ElasticNet":
                        results = modele_ridge_lasso_elastic(X_train, X_test, Y_train, Y_test)
                        st.success("✅ Modèles Ridge/Lasso/ElasticNet entraînés")
                        for name, (model, Y_pred, rmse) in results.items():
                            st.write(f"{name.capitalize()} - RMSE: {rmse:.4f}")
                            all_models[name] = model
                        model = results["ridge"][0]
                            
                    elif model_choice == "Régression Polynomiale":
                        # Utiliser la valeur du slider sauvegardée
                        degree = st.session_state['model_params']['poly_degree']
                        model, Y_pred, rmse = modele_polynomial(X_train, X_test, Y_train, Y_test, degree)
                        st.success(f"✅ Régression Polynomiale (degré {degree}) - RMSE: {rmse:.4f}")
                        all_models["polynomial"] = model
                        
                    elif model_choice == "Arbre de Décision":
                        model, Y_pred, rmse = modele_decision_tree(X_train, X_test, Y_train, Y_test)
                        st.success(f"✅ Arbre de Décision - RMSE: {rmse:.4f}")
                        all_models["decision_tree"] = model
                        
                    elif model_choice == "Random Forest":
                        # Utiliser la valeur du slider sauvegardée
                        n_est = st.session_state['model_params']['rf_estimators']
                        model, Y_pred, rmse, r2 = modele_random_forest(X_train, X_test, Y_train, Y_test, n_est)
                        st.success(f"✅ Random Forest - RMSE: {rmse:.4f}, R²: {r2:.4f}")
                        all_models["random_forest"] = model
                        
                    elif model_choice == "Gradient Boosting":
                        model, Y_pred, rmse = modele_gradient_boosting(X_train, X_test, Y_train, Y_test)
                        st.success(f"✅ Gradient Boosting - RMSE: {rmse:.4f}")
                        all_models["gradient_boosting"] = model
                        
                    elif model_choice == "XGBoost":
                        model, Y_pred, rmse = modele_xgboost(X_train, X_test, Y_train, Y_test)
                        st.success(f"✅ XGBoost - RMSE: {rmse:.4f}")
                        all_models["xgboost"] = model
                        
                    elif model_choice == "LightGBM":
                        model, Y_pred, rmse = modele_lightgbm(X_train, X_test, Y_train, Y_test)
                        st.success(f"✅ LightGBM - RMSE: {rmse:.4f}")
                        all_models["lightgbm"] = model
                        
                    elif model_choice == "CatBoost":
                        model, Y_pred, rmse = modele_catboost(X_train, X_test, Y_train, Y_test)
                        st.success(f"✅ CatBoost - RMSE: {rmse:.4f}")
                        all_models["catboost"] = model
                        
                    elif model_choice == "SVR":
                        model, Y_pred, rmse = modele_svr(X_train, X_test, Y_train, Y_test)
                        st.success(f"✅ SVR - RMSE: {rmse:.4f}")
                        all_models["svr"] = model
                        
                    elif model_choice == "KNN":
                        # Utiliser la valeur du slider sauvegardée
                        neighbors = st.session_state['model_params']['knn_neighbors']
                        model, Y_pred, rmse = modele_knn(X_train, X_test, Y_train, Y_test, neighbors)
                        st.success(f"✅ KNN - RMSE: {rmse:.4f}")
                        all_models["knn"] = model
                    
                    # Sauvegarder le modèle principal et tous les modèles
                    st.session_state['last_model'] = model
                    st.session_state['last_predictions'] = Y_pred
                    st.session_state['last_rmse'] = rmse
                    st.session_state['last_model_name'] = model_choice
                    st.session_state['all_models'] = all_models
                    st.session_state['model_trained'] = True
                    
                    st.info(f"💾 Modèle sauvegardé avec {len(feature_names)} features")
                    
            except Exception as e:
                st.error(f"❌ Erreur lors de l'entraînement : {e}")
        
        # Afficher les paramètres actuels pour débogage
        st.markdown("---")
        with st.expander("🔍 Paramètres actuels (débogage)"):
            st.write(f"Modèle sélectionné: {model_choice}")
            st.write(f"Degré polynomial: {st.session_state['model_params']['poly_degree']}")
            st.write(f"Nombre de voisins KNN: {st.session_state['model_params']['knn_neighbors']}")
            st.write(f"Nombre d'arbres Random Forest: {st.session_state['model_params']['rf_estimators']}")
# Page Comparaison des Modèles
elif page == "Comparaison des Modèles":
    st.header("📈 Comparaison des Modèles")
    
    if st.button("Lancer la comparaison de tous les modèles"):
        try:
            if 'X_train' not in st.session_state:
                st.warning("⚠️ Données non préparées")
            else:
                X_train = st.session_state['X_train']
                X_test = st.session_state['X_test']
                Y_train = st.session_state['Y_train']
                Y_test = st.session_state['Y_test']
                
                with st.spinner("Comparaison de tous les modèles en cours..."):
                    # Liste pour stocker les résultats
                    results = []
                    
                    # Tester chaque modèle
                    models_to_test = [
                        ("Régression Linéaire", modele_linear_regression),
                        ("Arbre de Décision", modele_decision_tree),
                        ("Random Forest", modele_random_forest),
                        ("Gradient Boosting", modele_gradient_boosting),
                        ("XGBoost", modele_xgboost),
                        ("LightGBM", modele_lightgbm),
                    ]
                    
                    for model_name, model_func in models_to_test:
                        try:
                            if model_name == "Random Forest":
                                model, Y_pred, rmse, r2 = model_func(X_train, X_test, Y_train, Y_test)
                                results.append({
                                    'Modèle': model_name,
                                    'RMSE': rmse,
                                    'R²': r2
                                })
                            else:
                                model, Y_pred, rmse = model_func(X_train, X_test, Y_train, Y_test)
                                # Calculer R² pour les autres modèles
                                r2 = r2_score(Y_test, Y_pred)
                                results.append({
                                    'Modèle': model_name,
                                    'RMSE': rmse,
                                    'R²': r2
                                })
                        except Exception as e:
                            st.warning(f"⚠️ {model_name} a échoué: {e}")
                    
                    # Afficher les résultats
                    if results:
                        results_df = pd.DataFrame(results)
                        st.subheader("Résultats de la comparaison")
                        st.dataframe(results_df.sort_values('RMSE'))
                        
                        # Graphique comparatif
                        fig, ax = plt.subplots(figsize=(12, 6))
                        models = results_df['Modèle']
                        rmse_values = results_df['RMSE']
                        
                        bars = ax.bar(models, rmse_values, color='skyblue')
                        ax.set_title('Comparaison des RMSE par modèle')
                        ax.set_ylabel('RMSE')
                        plt.xticks(rotation=45)
                        
                        # Ajouter les valeurs sur les barres
                        for bar, value in zip(bars, rmse_values):
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                                   f'{value:.4f}', ha='center', va='bottom')
                        
                        st.pyplot(fig)
                    else:
                        st.warning("Aucun modèle n'a pu être évalué")
                
        except Exception as e:
            st.error(f"❌ Erreur lors de la comparaison : {e}")

# Page Analyse des Performances
elif page == "Analyse des Performances":
    st.header("📊 Analyse des Performances des Modèles")
    
    if 'X_train' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord préparer les données dans la section 'Préprocessing'")
    else:
        X_train = st.session_state['X_train']
        X_test = st.session_state['X_test']
        Y_train = st.session_state['Y_train']
        Y_test = st.session_state['Y_test']
        
        if st.button("🔍 Analyser les performances de tous les modèles"):
            with st.spinner("Analyse en cours... Cela peut prendre quelques minutes"):
                try:
                    # Comparer tous les modèles
                    resultats = comparer_tous_modeles(X_train, X_test, Y_train, Y_test)
                    
                    if resultats:
                        # Créer un DataFrame avec les résultats
                        df_resultats = pd.DataFrame(resultats)
                        
                        # Afficher le classement
                        st.subheader("🏆 Classement des Modèles (du meilleur au pire)")
                        
                        # Style le tableau
                        styled_df = df_resultats[['Modèle', 'RMSE', 'R²']].style\
                            .format({'RMSE': '{:.4f}', 'R²': '{:.4f}'})\
                            .background_gradient(subset=['RMSE'], cmap='RdYlGn_r')\
                            .background_gradient(subset=['R²'], cmap='RdYlGn')
                        
                        st.dataframe(styled_df)
                        
                        # Sauvegarder le meilleur modèle
                        meilleur_modele = resultats[0]
                        st.session_state['meilleur_modele'] = meilleur_modele['Modèle_Objet']
                        st.session_state['meilleur_modele_nom'] = meilleur_modele['Modèle']
                        st.session_state['meilleur_rmse'] = meilleur_modele['RMSE']
                        st.session_state['meilleur_r2'] = meilleur_modele['R²']
                        
                        st.success(f"🎯 **Meilleur modèle identifié : {meilleur_modele['Modèle']}**")
                        st.info(f"📊 Performance : RMSE = {meilleur_modele['RMSE']:.4f}, R² = {meilleur_modele['R²']:.4f}")
                        
                        # Graphiques de comparaison
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Graphique RMSE
                            fig_rmse, ax_rmse = plt.subplots(figsize=(10, 6))
                            modeles = [r['Modèle'] for r in resultats]
                            rmse_values = [r['RMSE'] for r in resultats]
                            
                            bars = ax_rmse.bar(modeles, rmse_values, color=['green' if i == 0 else 'lightblue' for i in range(len(modeles))])
                            ax_rmse.set_title('Comparaison des RMSE\n(Plus bas = meilleur)')
                            ax_rmse.set_ylabel('RMSE')
                            plt.xticks(rotation=45, ha='right')
                            
                            # Ajouter les valeurs
                            for bar, value in zip(bars, rmse_values):
                                ax_rmse.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_values)*0.01, 
                                           f'{value:.4f}', ha='center', va='bottom', fontsize=8)
                            
                            st.pyplot(fig_rmse)
                        
                        with col2:
                            # Graphique R²
                            fig_r2, ax_r2 = plt.subplots(figsize=(10, 6))
                            r2_values = [r['R²'] for r in resultats]
                            
                            bars = ax_r2.bar(modeles, r2_values, color=['green' if i == 0 else 'lightblue' for i in range(len(modeles))])
                            ax_r2.set_title('Comparaison des R²\n(Plus haut = meilleur)')
                            ax_r2.set_ylabel('R²')
                            ax_r2.set_ylim([min(r2_values) - 0.1, 1.0])
                            plt.xticks(rotation=45, ha='right')
                            
                            # Ajouter les valeurs
                            for bar, value in zip(bars, r2_values):
                                ax_r2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                                         f'{value:.4f}', ha='center', va='bottom', fontsize=8)
                            
                            st.pyplot(fig_r2)
                        
                        # Conseils selon le meilleur modèle
                        st.subheader("💡 Recommandations")
                        meilleur_nom = meilleur_modele['Modèle']
                        
                        if "Random Forest" in meilleur_nom or "Gradient" in meilleur_nom or "XGBoost" in meilleur_nom:
                            st.info("**Forêts Aléatoires / Boosting recommandés :**\n"
                                  "- Données complexes avec relations non-linéaires\n"
                                  "- Bonne résistance au surapprentissage\n"
                                  "- Importance des features disponible")
                        elif "Linéaire" in meilleur_nom or "Ridge" in meilleur_nom or "Lasso" in meilleur_nom:
                            st.info("**Modèles Linéaires recommandés :**\n"
                                  "- Relations linéaires dans les données\n"
                                  "- Interprétabilité importante\n"
                                  "- Dataset de taille modérée")
                        elif "SVR" in meilleur_nom:
                            st.info("**SVR recommandé :**\n"
                                  "- Dataset de petite à moyenne taille\n"
                                  "- Frontières de décision complexes\n"
                                  "- Données normalisées")
                        else:
                            st.info("**Modèle sélectionné :**\n"
                                  "- Bonnes performances générales\n"
                                  "- À utiliser selon le contexte métier\n"
                                  "- Vérifier la stabilité sur de nouvelles données")
                            
                    else:
                        st.error("❌ Aucun modèle n'a pu être évalué")
                        
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'analyse : {e}")

# Page Prédiction
elif page == "Prédiction":
    st.header("🔮 Prédiction sur Nouvelles Données")
    
    if 'last_model' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord entraîner un modèle dans la section 'Modélisation'")
    else:
        # Sélection du modèle pour la prédiction
        st.subheader("Sélection du modèle pour la prédiction")
        
        # Options de sélection
        option_modele = st.radio(
            "Choisir le modèle à utiliser :",
            ["Dernier modèle entraîné", "Meilleur modèle identifié"]
        )
        
        if option_modele == "Meilleur modèle identifié" and 'meilleur_modele' in st.session_state:
            model = st.session_state['meilleur_modele']
            selected_model = st.session_state['meilleur_modele_nom']
            st.success(f"🎯 Utilisation du meilleur modèle : {selected_model}")
        else:
            # Sélection parmi les modèles disponibles
            available_models = []
            if 'all_models' in st.session_state:
                available_models = list(st.session_state['all_models'].keys())
            
            if available_models:
                selected_model_key = st.selectbox(
                    "Choisir le modèle à utiliser :",
                    available_models,
                    format_func=lambda x: {
                        'linear_regression': 'Régression Linéaire',
                        'ridge': 'Ridge',
                        'lasso': 'Lasso', 
                        'elastic': 'ElasticNet',
                        'polynomial': 'Régression Polynomiale',
                        'decision_tree': 'Arbre de Décision',
                        'random_forest': 'Random Forest',
                        'gradient_boosting': 'Gradient Boosting',
                        'xgboost': 'XGBoost',
                        'lightgbm': 'LightGBM',
                        'catboost': 'CatBoost',
                        'svr': 'SVR',
                        'knn': 'KNN'
                    }.get(x, x)
                )
                
                # Récupérer le modèle sélectionné
                model = st.session_state['all_models'][selected_model_key]
                selected_model = selected_model_key
                st.success(f"✅ Modèle {selected_model} sélectionné pour la prédiction")
            else:
                # Fallback sur le dernier modèle entraîné
                model = st.session_state['last_model']
                selected_model = st.session_state['last_model_name']
                st.info(f"ℹ️ Utilisation du dernier modèle entraîné: {selected_model}")
        
        st.subheader("Saisie des caractéristiques")
        
        # Récupérer les noms des features utilisées pour l'entraînement
        if 'feature_names' in st.session_state:
            feature_names = st.session_state['feature_names']
            st.info(f"📋 Le modèle attend {len(feature_names)} features")
            
            # Afficher les statistiques des features pour référence
            if 'X_original' in st.session_state:
                st.subheader("📊 Statistiques des features (pour référence)")
                stats_df = st.session_state['X_original'].describe()
                st.dataframe(stats_df)
            
            # Formulaire de saisie
            st.subheader("🎯 Saisie des valeurs pour la prédiction")
            input_data = {}
            
            # Créer 2 colonnes pour mieux organiser les inputs
            cols = st.columns(2)
            for i, feature in enumerate(feature_names):
                with cols[i % 2]:
                    # Obtenir les statistiques pour les placeholders
                    if 'X_original' in st.session_state:
                        min_val = float(st.session_state['X_original'][feature].min())
                        max_val = float(st.session_state['X_original'][feature].max())
                        mean_val = float(st.session_state['X_original'][feature].mean())
                        
                        input_data[feature] = st.number_input(
                            f"{feature}",
                            value=float(mean_val),
                            step=0.1,
                            help=f"Plage typique: {min_val:.2f} à {max_val:.2f}, Moyenne: {mean_val:.2f}"
                        )
                    else:
                        input_data[feature] = st.number_input(
                            f"{feature}",
                            value=0.0,
                            step=0.1
                        )
            
            if st.button("Faire une prédiction"):
                try:
                    # Créer le DataFrame d'entrée avec TOUTES les features dans le bon ordre
                    input_df = pd.DataFrame([input_data])[feature_names]
                    
                    st.info(f"📤 Données d'entrée: {len(feature_names)} features")
                    st.dataframe(input_df)
                    
                    # Standardiser les données d'entrée si nécessaire
                    if 'scaler' in st.session_state:
                        input_scaled = st.session_state['scaler'].transform(input_df)
                    else:
                        input_scaled = input_df
                    
                    # Prédiction
                    prediction_raw = model.predict(input_scaled)[0]
                    
                    # Convertir en 0 ou 1 avec un seuil
                    if prediction_raw >= 0.5:
                        prediction = 1
                    else:
                        prediction = 0
                    
                    # Afficher le résultat avec message personnalisé
                    st.success(f"**🎯 RÉSULTAT DE LA PRÉDICTION**")
                    
                    if prediction == 1:
                        st.error(f"**📊 Prédiction : 1 (RÉSILIATION)**")
                        st.info("💡 **Interprétation :** Selon les données fournies, le modèle prédit que le client va **RÉSILIER** son contrat.")
                    else:
                        st.success(f"**📊 Prédiction : 0 (NON-RÉSILIATION)**")
                        st.info("💡 **Interprétation :** Selon les données fournies, le modèle prédit que le client va **MAINTENIR** son contrat.")
                    
                    # Afficher aussi la valeur brute pour information
                    st.write(f"**Valeur brute du modèle :** {prediction_raw:.4f}")
                    
                    # Afficher des informations supplémentaires
                    if 'Y_original' in st.session_state:
                        y_min = st.session_state['Y_original'].min()
                        y_max = st.session_state['Y_original'].max()
                        st.info(f"📈 Plage des valeurs cibles dans les données: {y_min:.2f} à {y_max:.2f}")
                    
                except Exception as e:
                    st.error(f"❌ Erreur de prédiction : {e}")
                    st.info("💡 Assurez-vous que toutes les features sont correctement remplies")
        else:
            st.error("❌ Informations sur les features non disponibles")
# Footer
st.markdown("---")
st.markdown("Développé avec Streamlit • Utilisant le module de traitement ML complet")