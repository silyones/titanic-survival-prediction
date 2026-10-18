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
x_test = pd.read_csv("data/test/x_test.csv")

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

#training accuracy
train_log_model_pred = log_model.predict(x_train)
print("→ logistic regression training accuracy: ", accuracy_score(y_train,train_log_model_pred))

train_rf_model_pred = rf_model.predict(x_train)
print("→ random forest training accuracy: ", accuracy_score(y_train,train_rf_model_pred))

# evaluation metrics
print("→ logistic regression testing accuracy: ", accuracy_score(y_test,log_model_pred))
print("→ precision of logistic regression model: ", precision_score(y_test,log_model_pred))
print("→ recall of logistic regression model: ", recall_score(y_test,log_model_pred))
print("→ confusion matrix of logistic regression model: \n", confusion_matrix(y_test,log_model_pred))

print("→ random forest testing: ", accuracy_score(y_test,rf_model_pred))
print("→ precision of random forest model: ", precision_score(y_test,rf_model_pred))
print("→ recall of random forest model: ", recall_score(y_test,rf_model_pred))
print("→ confusion matrix of random forest model: \n", confusion_matrix(y_test,rf_model_pred))


'''
Model	               Training accuracy	Testing accuracy	Gap
Logistic Regression	   80.76%	            80.45%	            0.31 percentage points
Random Forest	       98.17%	            80.45%	            17.72 percentage points
'''