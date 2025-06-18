import pandas as pd

df = pd.read_csv("../data/users.csv") #a changer selon le chemin
if 'time_active' in df.columns:
    df = df.drop(columns=['time_active'])
anno_df = df.sample(n=200, random_state=42)
df_remaining = df.drop(anno_df.index)

anno_df['atypique'] = ''
anno_df['commentaire'] = ''

anno_df.to_csv("annotated_users.csv", index=False)

print("Fichiers exportés : train_users.csv (200 lignes), valid_users.csv (1000 lignes)")