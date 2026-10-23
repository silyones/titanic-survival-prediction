import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.metrics import classification_report, ConfusionMatrixDisplay



#load datasest
x_train = pd.read_csv("data/train/x_train.csv")
x_test = pd.read_csv("data/test/x_test.csv")

y_train = pd.read_csv("data/train/y_train.csv")["survived"]
y_test = pd.read_csv("data/test/y_test.csv")["survived"]

# load trained models
log_model = joblib.load("models/log_model.pkl")
rf_model = joblib.load("models/rf_modle.pkl")

# group models together using dictionary — name : model
models = {
    "logistic regresssion" : log_model,
    "random forest" : rf_model
}

#loop over each model: classification report
for name, model in models.items():
    print(f"{name}:\n")
    print(classification_report(
        y_test,
        model.predict(x_test),
        target_names = ["did not survive", "survived"]
    )) 

# confusion matrix for logistic regression
ConfusionMatrixDisplay.from_estimator(
    log_model,
    x_test,
    y_test,
    display_labels=["did not survive", "survived"]
)
plt.title("Confusion Matrix — Logistic Regression Model")
plt.savefig("plot/confusion_matrix_log_model")


# confusion matrix for random forest model
ConfusionMatrixDisplay.from_estimator(
    rf_model,
    x_test,
    y_test,
    display_labels=["did not survive","survived"]
)
plt.title("Confusion Matrix — Random Forest Model")
plt.savefig("plot/confusion_matrix_rf_model")
plt.show()

# random forest feature importance
importance = pd.Series(
    rf_model.feature_importances_,
    index = x_train.columns
).sort_values()

# print the importance score
print("Random Forest importance score:\n")
print(importance.sort_values(ascending=False).round(3))

'''OUTPUT:
Random Forest importance score:

fare          0.275
sex           0.265
age           0.251
pclass        0.085
sibsp         0.050
parch         0.039
embarked_S    0.023
embarked_Q    0.012
'''
# bar chart to show feature importance
sns.barplot(
    x = importance.values,
    y = importance.index
)
plt.xlabel("Importance score")
plt.ylable("Feature")
plt.title("Feature importance in Random Forest")
plt.savefig("plot/feature_importance.png")
plt.show()


"""
Iteration	name	              model
1	        Logistic Regression	  logistic_model
2	        Random Forest	      random_forest_model
""" 

