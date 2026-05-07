import pandas as pd
import os

def check_all_parquets():
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".parquet"):
                path = os.path.join(root, file)
                try:
                    df = pd.read_parquet(path)
                    if 'model' in df.columns:
                        models = df['model'].unique()
                        num_rows = len(df)
                        print(f"File: {path}")
                        print(f"  Rows: {num_rows}")
                        print(f"  Models: {models}")
                        if 'responses' in df.columns:
                            print(f"  Has responses column. First row responses count: {len(df.iloc[0]['responses'])}")
                except Exception as e:
                    pass

if __name__ == "__main__":
    check_all_parquets()
