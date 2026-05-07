import pandas as pd
import os

def find_responses():
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".parquet"):
                path = os.path.join(root, file)
                try:
                    df = pd.read_parquet(path)
                    if 'model' in df.columns and 'responses' in df.columns:
                        models = df['model'].unique()
                        print(f"File: {path}, Models: {models}, Rows: {len(df)}")
                except:
                    pass

if __name__ == "__main__":
    find_responses()
