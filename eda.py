from re import A
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

#make 6 plots in one fig
sns.set_style("whitegrid")
fig, axes = plt.subplots(2,3,figsize = (16,9))

# 1.countplot
sns.countplot(data=df,
x = "survived",
ax = axes[0,0])
axes[0,0].set_title("survived(0=no, 1 =yes)")

#survival by sex
sns.barplot(data=df,
x="sex",
y = "survived",
ax = axes[0,1])
axes[0,1].set_title("survival by sex")

# survival by class
sns.barplot(data= df,
x = "pclass",
y = "survived",
ax = axes[0,2])
axes.set_title("survival by class")

# age distribution 


