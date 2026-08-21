import pandas as pd
from pathlib import Path

def load_listing():

    csv_path = Path(__file__).resolve().parent.parent / "dataset" / "Resume.csv"
    df = pd.read_csv(csv_path, sep=',')

    data = df[['ID', 'Resume_str', 'Category']].copy()

    data = data.dropna(subset=["Resume_str", "Category"]).reset_index(drop=True)

    data = data.drop_duplicates(subset=["Resume_str", "Category"]).reset_index(drop=True)

    return data

