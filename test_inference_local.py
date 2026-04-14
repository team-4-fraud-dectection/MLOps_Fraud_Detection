import pandas as pd
import json

df = pd.read_parquet("data/featured/X_val.parquet")
samples = df.head(3).to_dict(orient="records")

with open("sample_request.json", "w", encoding="utf-8") as f:
    json.dump({"records": samples}, f, indent=2, default=float)

print("Saved 3 records to sample_request.json")