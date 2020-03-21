from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


dataset_path = "Iris Flower Classification\\dataset\\iris.csv"
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "flower_class"]
dataset = read_csv(dataset_path, names=columns)


data_array = dataset.values
X = data_array[:,0:4]
y = data_array[:, 4]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=1)


#I added 2 Linear and 4 Non-Linear models and compared their accuracy. 
classifying_models = []
classifying_models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
classifying_models.append(('LDA', LinearDiscriminantAnalysis()))
classifying_models.append(('KNN', KNeighborsClassifier()))
classifying_models.append(('CART', DecisionTreeClassifier()))
classifying_models.append(('NB', GaussianNB()))
classifying_models.append(('SVM', SVC(gamma='auto')))

model_results = []
model_names = []

for name, model in classifying_models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    model_results.append(cv_results)
    model_names.append(name)
    print('%s: %f (%f)' %(name, cv_results.mean(), cv_results.std()))


#This is the model implementation of non-linear algortihm. (Support Vector Machines)
model_0 = SVC(gamma='auto')
model_0.fit(X_train, Y_train)
predictions_0 = model_0.predict(X_test)

print(accuracy_score(Y_test, predictions_0))
print(confusion_matrix(Y_test, predictions_0))
print(classification_report(Y_test, predictions_0))


#Thisis the model implementation of linear algorithm. (Linear Discriminant Analysis)
model_1 = LinearDiscriminantAnalysis()
model_1.fit(X_train, Y_train)
predictions_1 = model_1.predict(X_test)

print(accuracy_score(Y_test, predictions_1))
print(confusion_matrix(Y_test, predictions_1))
print(classification_report(Y_test, predictions_1))
