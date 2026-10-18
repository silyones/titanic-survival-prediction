import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import(
    accuracy_score,
    precision_score,
    confusion_matrix,
    recall_score
)
import joblib 

#load datasest
x_train = pd.read_csv("data/train/x_train.csv")
x_test = pd.read_csv("data\test\x_test.csv")

y_train = pd.read_csv("data/train/y_train.csv")["survived"]
y_test = pd.read_csv("data/test/y_test.csv")["survived"]

#model 1: logistic regression 
log_model = LogisticRegression(
    max_iter = 1000
)

log_model.fit(x_train,y_train)

log_model_pred = log_model.predict(x_test)

# model 2: random forest
rf_model = RandomForestClassifier(
    n_estimators = 100,
    random_state = 42
)

rf_model.fit(x_train,y_train)

rf_model_pred = rf_model.predict(x_test)

# evaluation metrics
print("→ accuracy of logistic regression model: ", accuracy_score(y_test,log_model_pred))
print("→ precision of logistic regression model: ", precision_score(y_test,log_model_pred))
print("→ recall of logistic regression model: ", recall_score(y_test,log_model_pred))
print("→ confusion matrix of logistic regression model: \n", confusion_matrix(y_test,log_model_pred))

print("→ accuracy of random forest model: ", accuracy_score(y_test,rf_model_pred))
print("→ precision of random forest model: ", precision_score(y_test,rf_model_pred))
print("→ recall of random forest model: ", recall_score(y_test,rf_model_pred))
print("→ confusion matrix of random forest model: \n", confusion_matrix(y_test,rf_model_pred))
