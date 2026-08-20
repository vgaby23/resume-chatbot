import pandas as pd

def load_listing():

    df = pd.read_csv('Resume.csv', sep=',')

    data = df[['ID', 'Resume_str', 'Category']].copy()

    data = data.dropna(subset=["Resume_str", "Category"]).reset_index(drop=True)

    data = data.drop_duplicates(subset=["Resume_str", "Category"]).reset_index(drop=True)

    return data

