import pandas as pd
from sklearn.model_selection import train_test_split

#load the dataset
df = pd.read_csv("data/processed/clean_titanic.csv")

print("→ before encoding:\n", df.head())

#convert sex into numbers: M=0, F=1 — Mappling
df["sex"] = df["sex"].map({"male":0, "female":1})
print("→ sex mapping:\n", df["sex"])

# one hot encoding for embarmed column
df = pd.get_dummies(
    df,
    columns = ["embarked"],
    drop_first = True,
    dtype = int
)

print("→ after encoding:\n", df.head())

#seperate input(x) and target(y)
x = df.drop("survived", axis = 1)
y = df["survived"]

