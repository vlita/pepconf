import pandas as pd
import numpy as np

df = pd.read_pickle("pepconf_sampled.pkl")

print("=== Shape ===")
print(df.shape)

print("\n=== dtypes ===")
print(df.dtypes)

print("\n=== First row ===")
row = df.iloc[0]
for col in df.columns:
    val = row[col]
    if isinstance(val, np.ndarray):
        print(f"  {col}: ndarray {val.shape}\n{val[:3]}")
    else:
        print(f"  {col}: {val!r}")

print("\n=== Value counts: folder ===")
print(df.groupby("folder")["id"].nunique().rename("n_systems"))

print("\n=== n_atoms range ===")
print(df.groupby("id")["n_atoms"].first().describe())

print("\n=== basis / method coverage ===")
print("basis sets :", sorted(df["basis"].unique()))
# print("methods    :", sorted(df["method"].unique()))
