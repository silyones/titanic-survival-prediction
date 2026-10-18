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

#create a split
x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size = 0.2,
    random_state = 42,
    stratify = y
)

print("→ train shape: ", x_train.shape)
print("→ survival rate in y_train: ", y_train.mean())
print("→ survival rate in y_test: ", y_test.mean())

#save data
x_train.to_csv("data/train/x_train.csv", index=False)
y_train.to_csv("data/train/y_train.csv", index=False)

x_test.to_csv("data/test/x_test.csv", index=False)
y_test.to_csv("data/test/y_test.csv", index=False)
