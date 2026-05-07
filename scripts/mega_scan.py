import pandas as pd
import os

def scan_files():
    for root, dirs, files in os.walk("/"):
        if any(x in root for x in ["/proc", "/sys", "/dev", "/var/lib/docker"]):
            continue
        for file in files:
            if file.endswith((".parquet", ".csv")):
                path = os.path.join(root, file)
                try:
                    if file.endswith(".parquet"):
                        df = pd.read_parquet(path)
                    else:
                        df = pd.read_csv(path)

                    if "model" in df.columns:
                        models = df["model"].unique()
                        if len(models) >= 3:
                            print(f"FOUND INTERESTING FILE: {path}")
                            print(f"  Rows: {len(df)}")
                            print(f"  Models: {models}")
                except:
                    pass

if __name__ == "__main__":
    scan_files()
