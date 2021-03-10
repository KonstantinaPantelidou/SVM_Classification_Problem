import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns; sns.set()
from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import time
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import cross_val_score

#The dataset will only retain pictures of people that have at least 60 different pictures
faces=fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)

fig,ax=plt.subplots(4,4)
for i,axi in enumerate(ax.flat):
    axi.imshow(faces.images[i],cmap='bone')
    axi.set(xticks=[],yticks=[],xlabel=faces.target_names[faces.target[i]])
plt.show()

#keep 150 faces as a sample for 90% variance
#whiten for the features to be less corellated with each other and have the same variance
pca=PCA(n_components=150,whiten=True,random_state=42)

#If class_weight not given, all classes have weight one. 
svc=SVC(kernel='rbf', C=5, gamma=0.001, class_weight='balanced')

#Construct a Pipeline from the given estimators
model=make_pipeline(pca,svc)

X = faces.data
y = faces.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


hyper_params={'svc__C':[5,10],
			'svc__gamma':[1e-2, 1e-3, 1e-4]}

grid=GridSearchCV(model,hyper_params,cv=5)

grid.fit(X_train,y_train)

start = time.time()
model.fit(X_train, y_train)
stop = time.time()
print(f"Training time: {stop - start}s")

print(" The best parameters are: ", grid.best_params_)
model=grid.best_estimator_


#Try predicting using the test dataset
y_pred=model.predict(X_test)
print("Accuracy Score - test dataset:", metrics.accuracy_score(y_test, y_pred), "\n")

#Try predicting using the train dataset
y_predicted_train = model.predict(X_train)
print("Accuracy Score - train dataset:",metrics.accuracy_score(y_train, y_predicted_train) )

'''
#Stacking subplots in two directions
#by using the axes.flatten() method, we don’t have to go through the 
#hastle of nested for loops to deal with a variable number of rows and columns in our figure.
fig, ax=plt.subplots(4,6)
for i,axi in enumerate(ax.flat):
    axi.imshow(X_test[i].reshape(62,47),cmap='bone')
    axi.set(xticks=[],yticks=[])
    axi.set_ylabel(faces.target_names[y_pred[i]].split()[-1],color='black' if y_pred[i]==y_test[i] else 'red')
fig.suptitle('predicted Names; Incorrect labels in red',size=14);
plt.show()

#If  annot True, write the data value in each cell.
#cbar: whether to draw a color bar. False
matrix=confusion_matrix(y_test,y_pred)
sns.heatmap(matrix.T,square=True,annot=True,fmt='d',cbar=False,xticklabels=faces.target_names,yticklabels=faces.target_names)
plt.xlabel('True Label')
plt.ylabel('Predicted label');
plt.show()

print(classification_report(y_test,y_pred,target_names=faces.target_names))

#KNN Classifier
score = []
for k in range(1, 9):
 print('Begin KNN with k=',k)
 clf = KNeighborsClassifier(n_neighbors=k)
 print("Train model")
 clf.fit(X_train, y_train)
 print("Compute predictions")
 y_predicted = clf.predict(X_test)
 accuracy = accuracy_score(y_test, y_predicted)
 score.append(accuracy)
 print("Accuracy: ",accuracy)
 print("Confusion Matrix:\n", confusion_matrix(y_test, y_predicted))


print(classification_report(y_predicted, y_test))


plt.plot(range(1,9), score)
plt.title('Determining the Optimal Number of Neighbors')
plt.xlabel('K - Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

######################################################

#NCC algorithm
nc = NearestCentroid()
nc.fit(X_train, y_train)
 
score = nc.score(X_train, y_train)
print("Score: ", score)

cv_scores = cross_val_score(nc, X_train, y_train, cv=10)
print("CV average score: %.2f" % cv_scores.mean())

y_pred = nc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n",cm)

cr = classification_report(y_test, y_pred)
print(cr) 
'''