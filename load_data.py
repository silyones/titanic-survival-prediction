import seaborn as sns

# load dataset (0=no, 1=yes)
df = sns.load_dataset("titanic")

# first few rows
print("→ first few rows:\n", df.head())

# print shape
print("→shape:\n", df.shape)

# print columns
print("→columns:\n", df.columns)

# print info
print("→info:\n",df.info())

# print description
print("→description:\n", df.describe())

# check for null
print("→number of missing values:\n", df.isnull().sum())

# count duplicates
print("→number of duplicates\n", df.duplicated().sum())

# number of people that survived
print("→number of people who survived:\n", df["survived"].value_counts())

# save the dataset
df.to_csv("data/raw/titanic.csv", index=False)
print("saved dataset")