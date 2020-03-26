import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

titanic_df = pd.read_csv("train.csv")

def getting_substrings(main_string, substrings):
    for substring in substrings:
        if main_string.find(substring) != -1:
            return substring
    print(main_string)
    return np.nan


def replace_titles(x):
    title = x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Master']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title == 'Dr':
        if x['Sex'] == 'Male':
            return 'Mrs'
        else:
            return 'Mrs'
    else:
        return title

title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev', 'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess', 'Don', 'Jonkheer']

titanic_df['Title'] = titanic_df['Name'].map(lambda x: getting_substrings(x, title_list))

titanic_df['Title'] = titanic_df.apply(replace_titles, axis=1)

titanic_df['Family count'] = titanic_df['SibSp'] + titanic_df['Parch']


y = titanic_df['Survived']


features = ['Sex', 'Pclass', 'Family count', 'Age', 'Fare']
X = pd.get_dummies(titanic_df[features])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state = 1)


models = []

models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(max_depth=8, max_features='auto')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('NB', GaussianNB()))
models.append(('RFC', RandomForestClassifier(n_estimators=100, max_depth=8)))

model_names = []
model_performace = []

for name, model in models:
    Kfold = StratifiedKFold(n_splits=10, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=Kfold, scoring='accuracy')
    model_names.append(name)
    model_performace.append(cv_results)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

