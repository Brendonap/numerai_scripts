'https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling'

import pandas as pd
import numpy as np

from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from xgboost import XGBClassifier
from sklearn import metrics

import seaborn as sns
import matplotlib.pyplot as plt

def get_train_test_data(Round, is_test=False):

    prediction_data_set = Round + '/numerai_tournament_data.csv'
    training_data_set = Round + '/numerai_training_data1.csv'

    training_data = pd.read_csv(training_data_set, header=0)
    prediction_data = pd.read_csv(prediction_data_set, header=0)
    
    if (is_test == True):
        prediction_data = prediction_data[prediction_data["data_type"] == 'validation']

    features = [f for f in list(training_data) if "feature" in f]
    x_train = training_data[features]
    y_train = training_data["target"]

    x_test = prediction_data[features]
    y_test = prediction_data["target"]

    return x_train, y_train, x_test, y_test


def train_model(name, clf, param_grid, X, Y):

    print("begin training of " + name)
    model = clf

    grid_search_model = GridSearchCV(model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_log_error', n_jobs=4)
    grid_search_model.fit(X, Y)

    print("training complete")    
    return grid_search_model.best_estimator_


Round = 'Round110'
is_test = True

print("Get and split data")
x_train, y_train, x_test, y_test = get_train_test_data(Round, is_test)


# train all the base estimators on the training data
# we get the best estimator from each model and use a voting system to determine the final probability

print("training base models...")
# XGB Classifier
xgb_param_grid = {'n_estimators': [10]}
xgb_best = train_model('Xgboost', XGBClassifier(), xgb_param_grid, x_train, y_train)

# AdaBoost Classifier
ada_param_grid = {'n_estimators': [10]}
ada_best = train_model('Adaboost', AdaBoostClassifier(), ada_param_grid, x_train, y_train)

# Random Forrest Classifier
rf_param_grid = {'n_estimators': [10]}
rf_best = train_model('Random Forest', RandomForestClassifier(), rf_param_grid, x_train, y_train)

# Gradient Boost Classifier
#gb_param_grid = {'n_estimators': [10]}
#gb_best = train_model('Gradient Boost', GradientBoostingClassifier(), gb_param_grid, x_train, y_train)

if (is_test):
    test_Survived_xgb = pd.Series(xgb_best.predict(x_test), name="xgb")
    test_Survived_ada = pd.Series(ada_best.predict(x_test), name="ada")
    test_Survived_rf = pd.Series(rf_best.predict(x_test), name="rf")
    #test_Survived_gb = pd.Series(gb_best.predict(x_test), name="gb")

    #ensemble_results = pd.concat([test_Survived_xgb,test_Survived_ada,test_Survived_rf,test_Survived_gb],axis=1)
    ensemble_results = pd.concat([test_Survived_xgb,test_Survived_ada,test_Survived_rf],axis=1)
    g = sns.heatmap(ensemble_results.corr(),annot=True)
    plt.show(g)
    #g.savefig("heatmap.png")


print("combining base models...")
# Combine Estimators
#votingC = VotingClassifier(estimators=[('xgb', xgb_best), ('ada', ada_best), ('rf', rf_best), ('gb', gb_best)], voting='soft', n_jobs=4, weights=[5,1,1,1])
votingC = VotingClassifier(estimators=[('xgb', xgb_best), ('ada', ada_best), ('rf', rf_best)], voting='soft', n_jobs=4, weights=[5,1,1])
votingC.fit(x_train, y_train)
voting_prediction = votingC.predict_proba(x_test)

if (is_test):
    log_loss_result = metrics.log_loss(y_test, voting_prediction)
    print("log loss result")
    print(log_loss_result)



# TODO: 
# 1. fill in more vlues for gridsearch to look over, creating better fits
# 3. include more models
# 4. find away to efficiently compare correlations of models and perhaps dynamically choose the n best models
# 5. try out diffent loss/scoring functions
# 6. try use geometric mean for ensamble
# 7. PCA for feature engineering

# thoughts for v5+
# find a different wy of combining estimators (bagging/stacking/boosting)
# start looking into possiblity of feature engineering