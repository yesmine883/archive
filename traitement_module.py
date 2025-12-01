# traitement_module.py - VERSION RÉGRESSION

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==========================
# Chargement et préparation
# ==========================

def charger_donnees(fichier_csv):
    """Charge un CSV avec encodage latin1 et séparateur ;"""
    df = pd.read_csv(fichier_csv, sep=';', encoding='latin1')
    return df

def encoder_colonnes(X, colonnes):
    """Encode les colonnes catégorielles avec LabelEncoder"""
    le = LabelEncoder()
    for c in colonnes:
        if c in X.columns:
            X[c] = le.fit_transform(X[c])
    return X

def split_scale(X, Y, test_size=0.2):
    """Sépare en train/test et standardise les features"""
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=0)
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)
    return X_train_sc, X_test_sc, Y_train, Y_test

# ==========================
# Modèles de RÉGRESSION
# ==========================

def modele_linear_regression(X_train, X_test, Y_train, Y_test):
    model = LinearRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    return model, Y_pred, rmse

def modele_ridge_lasso_elastic(X_train, X_test, Y_train, Y_test, alpha_ridge=1.0, alpha_lasso=0.01, alpha_elastic=0.1, l1_ratio=0.5):
    ridge_model = Ridge(alpha=alpha_ridge)
    ridge_model.fit(X_train, Y_train)
    Y_ridge = ridge_model.predict(X_test)

    lasso_model = Lasso(alpha=alpha_lasso)
    lasso_model.fit(X_train, Y_train)
    Y_lasso = lasso_model.predict(X_test)

    elastic_model = ElasticNet(alpha=alpha_elastic, l1_ratio=l1_ratio)
    elastic_model.fit(X_train, Y_train)
    Y_elastic = elastic_model.predict(X_test)

    rmse_ridge = np.sqrt(mean_squared_error(Y_test, Y_ridge))
    rmse_lasso = np.sqrt(mean_squared_error(Y_test, Y_lasso))
    rmse_elastic = np.sqrt(mean_squared_error(Y_test, Y_elastic))

    return {
        "ridge": (ridge_model, Y_ridge, rmse_ridge),
        "lasso": (lasso_model, Y_lasso, rmse_lasso),
        "elastic": (elastic_model, Y_elastic, rmse_elastic)
    }

def modele_polynomial(X_train, X_test, Y_train, Y_test, degree=2):
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_poly, Y_train)
    Y_pred = model.predict(X_test_poly)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    return model, Y_pred, rmse

def modele_decision_tree(X_train, X_test, Y_train, Y_test):
    model = DecisionTreeRegressor(random_state=0)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    return model, Y_pred, rmse

def modele_random_forest(X_train, X_test, Y_train, Y_test, n_estimators=100):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=0)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    r2 = r2_score(Y_test, Y_pred)
    return model, Y_pred, rmse, r2

def modele_gradient_boosting(X_train, X_test, Y_train, Y_test, n_estimators=100, learning_rate=0.1, max_depth=3):
    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=0)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    return model, Y_pred, rmse

def modele_xgboost(X_train, X_test, Y_train, Y_test, n_estimators=100, learning_rate=0.1, max_depth=3):
    model = XGBRegressor(objective='reg:squarederror', n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=0)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    return model, Y_pred, rmse

def modele_lightgbm(X_train, X_test, Y_train, Y_test, n_estimators=100, learning_rate=0.1, max_depth=3):
    model = lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=0)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    return model, Y_pred, rmse

def modele_catboost(X_train, X_test, Y_train, Y_test, iterations=500, learning_rate=0.1, depth=3):
    model = CatBoostRegressor(iterations=iterations, learning_rate=learning_rate, depth=depth, verbose=0, random_state=0)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    return model, Y_pred, rmse

def modele_svr(X_train, X_test, Y_train, Y_test, kernel='rbf', C=100, gamma=0.1, epsilon=0.1):
    model = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    return model, Y_pred, rmse

def modele_knn(X_train, X_test, Y_train, Y_test, n_neighbors=5):
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    return model, Y_pred, rmse

# ==========================
# Fonction de comparaison des modèles (RÉGRESSION)
# ==========================

def comparer_tous_modeles(X_train, X_test, Y_train, Y_test):
    """
    Compare tous les modèles de régression et retourne les résultats classés
    """
    resultats = []
    
    # Liste de tous les modèles à tester
    modeles_a_tester = [
        ("Régression Linéaire", modele_linear_regression),
        ("Ridge", lambda x_tr, x_te, y_tr, y_te: modele_ridge_lasso_elastic(x_tr, x_te, y_tr, y_te)["ridge"]),
        ("Lasso", lambda x_tr, x_te, y_tr, y_te: modele_ridge_lasso_elastic(x_tr, x_te, y_tr, y_te)["lasso"]),
        ("ElasticNet", lambda x_tr, x_te, y_tr, y_te: modele_ridge_lasso_elastic(x_tr, x_te, y_tr, y_te)["elastic"]),
        ("Arbre de Décision", modele_decision_tree),
        ("Random Forest", modele_random_forest),
        ("Gradient Boosting", modele_gradient_boosting),
        ("XGBoost", modele_xgboost),
        ("LightGBM", modele_lightgbm),
        ("SVR", modele_svr),
        ("KNN", modele_knn)
    ]
    
    for nom_modele, fonction_modele in modeles_a_tester:
        try:
            if nom_modele == "Random Forest":
                modele, Y_pred, rmse, r2 = fonction_modele(X_train, X_test, Y_train, Y_test)
                resultats.append({
                    'Modèle': nom_modele,
                    'RMSE': rmse,
                    'R²': r2,
                    'Modèle_Objet': modele
                })
            else:
                modele, Y_pred, rmse = fonction_modele(X_train, X_test, Y_train, Y_test)
                # Calculer R² pour les autres modèles
                r2 = r2_score(Y_test, Y_pred)
                resultats.append({
                    'Modèle': nom_modele,
                    'RMSE': rmse,
                    'R²': r2,
                    'Modèle_Objet': modele
                })
                
        except Exception as e:
            print(f"❌ {nom_modele} a échoué: {e}")
    
    # Trier par RMSE (meilleur = plus faible)
    resultats_tries = sorted(resultats, key=lambda x: x['RMSE'])
    
    return resultats_tries