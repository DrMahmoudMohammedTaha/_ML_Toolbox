
# Cross Validation Classification Accuracy
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

dataframe = pd.read_csv(data.csv)
array = dataframe.values
X = Predictors
Y = Response # variable
kfold = model_selection.KFold(n_splits=10, random_state=4)
model = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Accuracy: %.3f (%.3f)") % (results.mean(), results.std())