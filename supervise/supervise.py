import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Charger les données annotées (anciennement annotated_users.csv)
df_train = pd.read_csv("supervise/annotated_users.csv")

# Filtrer les utilisateurs avec une annotation valide
df_train = df_train[df_train['atypique'].isin([0, 1, '0', '1'])].copy()
df_train['atypique'] = df_train['atypique'].astype(int)

# Définir les features utilisées pour l'entraînement
features = [
    'n_tweets', 'n_replies', 'n_quotes', 'n_retweets', 'max_followers',
    'n_mentions', 'n_hashtags', 'n_urls', 'n_out_urls', 'f_tweets', 'f_friends'
]

X = df_train[features]
y = df_train['atypique']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

mode = input("Choisissez le mode (1 = Validation, 2 = Application sur test): ").strip()

# VALIDATION
if mode == "1":
    df_val = df_train.sample(n=50, random_state=42)
    df_train_rest = df_train.drop(df_val.index)

    X_train = scaler.fit_transform(df_train_rest[features])
    y_train = df_train_rest['atypique']

    X_val = scaler.transform(df_val[features])
    y_val = df_val['atypique']


    model = SVC(kernel='rbf', C=1.0, gamma='scale')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    print("\n Résultats de validation ")
    print(confusion_matrix(y_val, y_pred))
    print(classification_report(y_val, y_pred))

# TEST
elif mode == "2":
    df_test = pd.read_csv("data/users.csv")

    known_ids = set(df_train['user_id'])
    df_test = df_test[~df_test['user_id'].isin(known_ids)].copy()

    if df_test.empty:
        print("Le fichier test ne contient aucun utilisateur nouveau.")
        exit()

    print(df_test.columns)
    X_test = df_test[features]
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel='rbf', C=1.0, gamma='scale')
    model.fit(X_scaled, y)

    preds = model.predict(X_test_scaled)
    df_test['atypique_pred'] = preds

    df_test.to_csv("predicted_users.csv", index=False)
    print(f"\nPrédictions sauvegardées dans 'predicted_users.csv' ({len(df_test)} utilisateurs)")
    
    # Nombre d'utilisateurs prédits
    df = pd.read_csv("predicted_users.csv")
    n_atypiques = (df['atypique_pred'] == 1).sum()
    print(f"Nombre d'utilisateurs prédits comme atypiques : {n_atypiques}")

else:
    print("Mode invalide. Tapez 1 ou 2.")