import pandas as pd

#load csv
df = pd.read_csv("data/raw/titanic.csv")

#check foro missing values
print("→missing values:\n", df.isnull().sum())

#drop unnecessary columns:
#deck:has too many missing values: 688