import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.metrics import classification_report
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import gzip
import random
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale

#MNIST Dataset was used and it was downloaded from Kaggle.com
#It can be found here ----> https://drive.google.com/drive/folders/1qsfntkfwAH3xMtu_eIdMtW7NRoTbBdm4?usp=sharing

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

print(train_data.shape)
print(test_data.shape)
print(train_data.head())
print(test_data.head())
print(train_data.isnull().sum().head(10))
print(test_data.describe())
print(train_data.describe())

# about the dataset

# dimensions
print("Dimensions: ",test_data.shape, "\n")
print(test_data.info())
print(test_data.head())

# dimensions
print("Dimensions: ",train_data.shape, "\n")
print(train_data.info())
print(train_data.head())

print(train_data.columns)
print(test_data.columns)


order = list(np.sort(train_data['label'].unique()))
print(order)

## Separating the X and Y variable

y = train_data['label']

## Dropping the variable 'label' from X variable 
X = train_data.drop(columns = 'label')

## Printing the size of data 
print(train_data.shape)

X = X/255.0
test_data = test_data/255.0

print("X:", X.shape)
print("test_data:", test_data.shape)

X_scaled = scale(X)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.4, train_size = 0.6 ,random_state = 10)

print(X_train.shape)

'''

#K-Fold cross validation

folds = KFold(n_splits = 5, shuffle = True, random_state = 10)

# specify range of hyperparameters
# Set the parameters by cross-validation
hyper_params = [ {'gamma': [1e-2, 1e-3, 1e-4],
                     'C': [5,10]}]


# specify model
model = SVC(kernel="rbf")

# set up GridSearchCV()
model_cv = GridSearchCV(estimator = model, 
                        param_grid = hyper_params, 
                        scoring= 'accuracy', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True)      

# fit the model
model_cv.fit(X_train, y_train)

# cv results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results

# converting C to numeric type for plotting on x-axis
cv_results['param_C'] = cv_results['param_C'].astype('int')

# # plotting
plt.figure(figsize=(16,8))

# subplot 1/3
plt.subplot(131)
gamma_01 = cv_results[cv_results['param_gamma']==0.01]

plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"])
plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.01")
plt.ylim([0.60, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')

# subplot 2/3
plt.subplot(132)
gamma_001 = cv_results[cv_results['param_gamma']==0.001]

plt.plot(gamma_001["param_C"], gamma_001["mean_test_score"])
plt.plot(gamma_001["param_C"], gamma_001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.001")
plt.ylim([0.60, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')


# subplot 3/3
plt.subplot(133)
gamma_0001 = cv_results[cv_results['param_gamma']==0.0001]

plt.plot(gamma_0001["param_C"], gamma_0001["mean_test_score"])
plt.plot(gamma_0001["param_C"], gamma_0001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.0001")
plt.ylim([0.60, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')

# printing the optimal accuracy score and hyperparameters
best_score = model_cv.best_score_
best_hyperparams = model_cv.best_params_

print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))
'''
##################################################

# SVM - model
model = SVC(C=10, gamma=0.001, kernel="rbf")
start = time.time()
model.fit(x_train, y_train)
stop = time.time()
print(f"Training time: {stop - start}s")

y_pred = model.predict(x_test)
print("Accuracy Score - test dataset:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")

y_predicted_train = model.predict(x_train)
print("Accuracy Score - train dataset:",metrics.accuracy_score(y_train, y_predicted_train) )

#print(metrics.accuracy_score(y_train, y_predicted_train))
print(metrics.confusion_matrix(y_test, y_pred), "\n")

print(classification_report(y_pred, y_test))


# Nearest Centroid Classifier

nc = NearestCentroid()
nc.fit(x_train, y_train)
 
score = nc.score(x_train, y_train)
print("Score: ", score)

cv_scores = cross_val_score(nc, x_train, y_train, cv=10)
print("CV average score: %.2f" % cv_scores.mean())

y_pred = nc.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n",cm)

cr = classification_report(y_test, y_pred)
print(cr)

##################################################

# KNN Classifier

score = []
for k in range(1, 9):
 print('Begin KNN with k=',k)
 clf = KNeighborsClassifier(n_neighbors=k)
 print("Train model")
 clf.fit(x_train, y_train)
 print("Compute predictions")
 y_predicted = clf.predict(x_test)
 accuracy = accuracy_score(y_test, y_predicted)
 score.append(accuracy)
 print("Accuracy: ",accuracy)
 print("Confusion Matrix:\n", confusion_matrix(y_test, y_predicted))


plt.plot(range(1,9), score)
plt.title('Determining the Optimal Number of Neighbors')
plt.xlabel('K - Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

clf = KNeighborsClassifier(n_neighbors=3)
print("Train model")
clf.fit(x_train, y_train)
print("Compute predictions")
y_predicted = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_predicted)
#score.append(accuracy)
print("Accuracy: ",accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_predicted))

print(classification_report(y_predicted, y_test))

##################################################
