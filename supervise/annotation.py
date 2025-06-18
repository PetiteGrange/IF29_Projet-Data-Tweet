import pandas as pd
from IPython.display import clear_output

df = pd.read_csv("annotated_users.csv")

expected_columns = {
    'user_id', 'n_tweets', 'n_replies', 'n_quotes', 'n_retweets', 'max_followers',
    'n_mentions', 'n_hashtags', 'n_urls', 'n_out_urls',
    'f_tweets', 'f_friends', 'atypique', 'commentaire'
}
if not expected_columns.issubset(df.columns):
    raise ValueError("Le fichier annotated_users.csv ne contient pas toutes les colonnes attendues.")

df['atypique'] = df['atypique'].astype(str)
to_annotate = df[df['atypique'].isin(['', 'nan'])].copy()

print(f"{len(to_annotate)} utilisateurs à annoter")

# Boucle d'annotation
for i, row in to_annotate.iterrows():
    clear_output()
    print(f"Utilisateur {row['user_id']} ({i+1}/{len(df)})\n")

    print("=== Statistiques utilisateur ===")
    print(f"- Tweets               : {row['n_tweets']}")
    print(f"- Retweets             : {row['n_retweets']}")
    print(f"- Réponses             : {row['n_replies']}")
    print(f"- Citations            : {row['n_quotes']}")
    print(f"- Max followers vus    : {row['max_followers']}")
    print(f"- Mentions             : {row['n_mentions']}")
    print(f"- Hashtags             : {row['n_hashtags']}")
    print(f"- URLs                 : {row['n_urls']}")
    print(f"- URLs externes        : {row['n_out_urls']}")
    print(f"- Fréquence des tweets : {row['f_tweets']:.4f}")
    print(f"- Fréquence d'amis   ' : {row['f_friends']:.4f}\n")

    while True:
        atypique = input("Ce compte est-il atypique ? (1 = Oui, 0 = Non) : ").strip()
        if atypique in ['0', '1']:
            df.at[i, 'atypique'] = atypique
            break
        print("Réponse invalide. Tapez 1 pour Oui ou 0 pour Non.")

    commentaire = input("Commentaire (laisser vide si aucun) : ").strip()
    df.at[i, 'commentaire'] = commentaire
    df.to_csv("annotated_users.csv", index=False)
    print(f"Annotation enregistrée pour l'utilisateur {row['user_id']}")

clear_output()
print("Annotation terminée pour tous les utilisateurs.")