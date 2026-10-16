import pandas as pd

#load csv
df = pd.read_csv("data/raw/titanic.csv")

#check foro missing values
print("→missing values:\n", df.isnull().sum())

#drop unnecessary columns:
#    → deck:has too many missing values: 688
#    → are derived from other columns: who, adult_male, alone, alive, class
df = df.drop(columns=["who","adult_male","alone","alive","deck","class","embark_town"])

#check missing columns after cleaning
print("→ columns after cleaning:\n", df.isnull().sum())

#fill missing age with median age
df["age"] = df["age"].fillna(df["age"].median())

#use mode to fill embarked
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

#check missing columns after cleaning
print("→ columns after cleaning:\n", df.isnull().sum())

