import pandas as pd
import matplotlib.pyplot as plt
import joblib
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



"""
Iteration	name	              model
1	        Logistic Regression	  logistic_model
2	        Random Forest	      random_forest_model
""" 

