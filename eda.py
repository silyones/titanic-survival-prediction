import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load dataset
df = pd.read_csv("data/processed/clean_titanic.csv")

#overall survival rate
print("→ overall suvival rate:", round(df["survived"].mean()*100,1))

#survival rate by sex
print("→ survival rate by sex: ", df.groupby("sex")["survived"].mean().round(2))

#survival rate by pclass
print("→ survival rate by pclass: ", df.groupby("pclass")["survived"].mean().round(2))

#survival rate by embarked
print("→ survival rate by pclass: ", df.groupby("embarked")["survived"].mean().round(2))
