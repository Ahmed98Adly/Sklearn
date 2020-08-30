# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:57:39 2020

@author: AA
"""

#IMPORTING THE LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import the dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


#TAKE CARE OF MISSING DATA
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy= 'mean')
imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])


#Encoding categorical Data (FOR MANY CATEGOREIS [ MORE THAN 2])
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers= [( 'encoder' , OneHotEncoder(), [0])], remainder = 'passthrough')
x = ct.fit_transform(x)

#Encoding Dependent variable (FOR 2 CATEGROIES [BUNARY])
from sklearn.preprocessing import LabelEncoder
Le = LabelEncoder()
y = Le.fit_transform(y)


#Split dataset into Training and Testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2 , random_state = 1)


#Feature Scaling (AFTER SPLITTING DATASET) BY STANDARISATION =  (X- MEAN)/ STANDADR DEVIATION
from sklearn.preprocessing import StandardScaler
Sc = StandardScaler()
x_train[:,3:] = Sc.fit_transform(x_train[:,3:])

#Only Transform for x_ test to have the same scaler as training set and not to get different scaler
x_test[:,3:] = Sc.transform(x_test[:,3:]) 



''Linear Regression''
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#split train test dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)


#import LinearRegression and fit training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


#predict test test
y_pred = regressor.predict(x_test)


#visualise training set
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('salaries vs experience(training set) ')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

#visualise test set
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('salaries vs experience(test set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()


-SUPPORT VECTOR REGRESSION SVR-
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#change Y to 2D array so we can apply feature scalling
y = y.reshape(len(y), 1)


#Feature Scalling for matrix of features (X)
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)

"we need different features scaling for x and y ( bec. different mean and range of numbers)"

#Feature of scalling for dependent variable (y)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)


#we will not spli data as small number


#Import SVR AND train the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x, y)

#predict new results and get back to original values ( get back from featuring scaling)
s = regressor.predict(sc_x.transform([[6.5]]))
sc_y.inverse_transform(s)


#visuale SVR Results
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x)), color = 'blue')
plt.title('salaries vs level')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()

#visualising SVR result (for high resolution and smoother  curve)
x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid))), color = 'blue')
plt.title('salaries vs level')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()


"Decision trees" - (not good for using for small number of variables in the split as no change in mean)
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[: , 1:-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y), 1)

#train the whole dataset for decision trees
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x, y)


#predict a new result
y_pred = regressor.predict([[6.5]])

#visualisa the results in high resolution
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('salaries vs level')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()


"Rnadom forest Regression"  
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y), 1)

#training dataset on random forest regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)    -n_estimators are number of trees-
regressor.fit(x, y)

#predict a new result
regressor.predict([[6.5]])


#visualise the results in high resolution
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('salaries vs level')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()


Evaulation of regressor models by ADJUSTED R^2 , closer to 1 then better model
#EVALUATION THE MODEL PERFORMANCE
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


-CLASSIFICATION-
'LOGISTIC REGRESSION'
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('breast_cancer_weka_dataset.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

y.reshape(len(y), 1)

#split dataset to training and test datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,  y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#training the logistic regression on the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

#predict a new results
y_pred = classifier.predict(sc.transform(([[30, 87000]])))

#predict the test set results
y_pred_test = classifier.predict(x_test)
np.concatenate((y_pred_test.reshape(len(y_pred_test), 1), y_test.reshape(len(y_test),1)), 1)

#making the cpnfusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred_test)
ac = accuracy_score(y_test, y_pred_test)

#NOT IMPORTANT
#visualise training set results ONLY POSSIBLE IF WE HAVE 2 FEATURES(X,Y) EACH ON AN AXIS
#visualise test set results


"K-Nearest neighbour" non linear classifier
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y), 1 )

#split dataset to training and test datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,  y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)



#training the KNN on the training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(metric = 'minkowski', p = 2 ) to use euclidean distance
classifier.fit(x_train, y_train)

#predict test set
y_pred = classifier.predict(sc.transform(x_test))
np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test),1)), 1)

#making the cpnfusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)

"support vector machine"
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y), 1 )

#split dataset to training and test datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,  y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


#training the SVM on the training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(x_train, y_train)

#predict test set
y_pred = classifier.predict(x_test)
np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test),1)), 1)

#making the cpnfusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)

"KERNEL SVM" FEATURE SCALLING IS IMPORTANT FOR ACCURACY
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y), 1 )

#split dataset to training and test datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,  y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)



#training the SVM on the training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train, y_train)

#predict test set
y_pred = classifier.predict(x_test)
np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test),1)), 1)

#making the cpnfusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)


CLASSIFICATION
"HEART DISEASE PREDICTION"
#import libraries
import numpy as np
import pandas as pd

#import dataset
dataset = pd.read_csv('heart_disease_weka_dataset.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#split into train and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2 , random_state = 0)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(metric = 'minkowski', p = 2 ) to use euclidean distance
classifier.fit(x_train, y_train)

#train the train set by 
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier((metric = 'minkowski', p = 2 ))
classifier.fit(x_train, y_train)

#predict test results
y_pred = classifier.predict(x_test)


#check accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)


"NAIVE BAYES"
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y), 1 )

#split dataset to training and test datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,  y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#training the naive bayes on the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

#predict test set
y_pred = classifier.predict(x_test)
np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test),1)), 1)

#making the cpnfusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)


"Decision trees Classification"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y), 1 )

#split dataset to training and test datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,  y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#training the decision trees on the training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

#predict test set
y_pred = classifier.predict(x_test)
np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test),1)), 1)

#making the cpnfusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)


"Random Forest Classification"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y), 1 )

#split dataset to training and test datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,  y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#training the random forest on the training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators =10 , criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

#predict test set
y_pred = classifier.predict(x_test)
np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test),1)), 1)

#making the cpnfusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)


"kmeans clustering"
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:,3:].values
"just choose 2 columns to be able to visualise at the end in 2D graph"

"we should choose all except cusID as all interfere in finding the clusters"

"no DEPENDENT VARIABLE (Y)" , JUST SEARCHING FOR CLUSTERS ( FEATURES IN THE DATASET )

#using ELBOW METHOD to find optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.xlabel('number of clusters')
plt.ylabel('wcss')
plt.title('The elbow method to find the best number of clusters')
plt.show()   

" Best number of clusters is 5" 

#training the K means on the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(x)

"fit_predict to show the clusters (dependent variable")

#visualise the clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'cluster 3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'cluster 4')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'cluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c ='yellow' , label = 'centroids')
plt.title('clusters of customers')
plt.xlabel('Annual income')
plt.ylabel('spending score')
plt.legend()
plt.show()


"we can train the kmeans with more than 2 features BUT we wont be able to visualise"
--------------------------------------------

"Hierarchical Clustering"

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, 3:].values

#using DENDROGRAM to find optimal number of cluster
from scipy.cluster import hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.title('dendrogram')
plt.xlabel('observation points(customers)')
plt.ylabel('Euclidean distance')
plt.show()

"therefore from the dendrogram, the best number of clusters is 5"

#training the hierarchical clustering (agglomerative)on the dataset
from sklearn.cluster import AgglomerativeClustering
ag = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_kmeans= ag.fit_predict(x)

#visualise the clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'cluster 3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'cluster 4')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'cluster 5')
plt.title('clusters of customers')
plt.xlabel('Annual income')
plt.ylabel('spending score')
plt.legend()
plt.show()


"Association rule learning"

!pip install apyori
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#import dataset 
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0,20)])
    
#train the aproiri on dataset
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

#visualise the results
results = list(rules)
results

#putting results well organised into a  pnadas dataframe


"ECLAT"
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#import dataset 
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0,20)])
    
#train the eclat on dataset
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

#visualise the results
results = list(rules)
results


"MULTIPLE LINEAR REGRESSION"
#imprt libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#dealing with dummy variables with onehotencoder
#encoding cateogiraldata
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers= [( 'encoder' , OneHotEncoder(), [3])], remainder = 'passthrough')
x = ct.fit_transform(x)


#split dataset to train and test dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#training multiple linear regression of training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predict test results
y_pred = regressor.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))

#evaluate model performance
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

"polynomial linear regression"
#imprt libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#train linear regression on dataset
from sklearn.linear_model import LinearRegression
linregressor = LinearRegression()
linregressor.fit(x, y)

#training polnomial regression on dataset
from sklearn.preprocessing import PolynomialFeatures
polyregressor = PolynomialFeatures(degree = 2)
x_poly = polyregressor.fit_transform(x)

linregressor2 = LinearRegression()
linregressor2.fit(x_poly, y)

#predict new salary with linear regression
y_pred = linregressor.predict([[6.5]])

#predict a new salary with polynomial
y_pred2 = linregressor2.predict(polyregressor.fit_transform([[6.5]]))




"Deep learning   ARTIFICIAL NEURAL NETWORK"
#import libraries
import numpy as np
import pandas as pd
import tensorflow as tf

#import dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:-1 ].values
y = dataset.iloc[:, -1].values

#encoding categorial data
#encode geography
#Encoding categorical Data (FOR MANY CATEGOREIS [ MORE THAN 2])
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers= [( 'encoder' , OneHotEncoder(), [1])], remainder = 'passthrough')
x = ct.fit_transform(x)

#encoding gender column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])


#split training and test set
from sklearn.model_selection import train_test_split
x_train,  x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2 , random_state = 0)



"feature scalling is a must in DEEP LEARNING"
#feature scalling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Buidling the ANN
#initilize the ANN "create ANN as sequence of layers"
ann = tf.keras.models.Sequential()

#add input library and hidden layer
"input layers are the input parameters( gender, geography, cashcard,.....)
"6 is number of hidden layers which is hyperparameter(no optimal value but just by experimenting)"
"relu is rectifier "
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))

#add second hiddent layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))

#add output layer
"units is 1 as we only want one output of binary results and the activation is sigmoid to give the probability to leave the bank"
ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))


#training the ANN
#compiling the ann
"adam use stochastic gradient descent( to update the weights therefore reduce loss function)"
"loss is to measure the loss function(how far is the predicted results from the right result)"
"binary_crossentropy in classification for binary outcome"
"category_crossentropy for more than a binary output"

ann.compile(optimizer= 'adam', loss = 'binary_crossentropy' , metrics = ['accuracy'])


#train the ANN on the training set
" as batch learning so we need batch size = for the stochastic gradient descent, default value is 32(it is hyperparameter)
"1 epoch is passing the whole training set one time"
ann.fit(x_train, y_train, batch_size = 32, epochs = 100)

#predict the results
"bec. of sigmoid we will get the probabilty
ypred = ann.predict(x_test)
" to give whether yes or no result
y_predd = ypred > 0.5

#predict new resulst
" [[]] as results need too be 2D array and to change the dummy variable[1,0,0 as france] and feature scalling"
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))

#evaluate accuracy
"ypredd as it is the (o, 1)while y_pred is the probability
from sklearn.metrics import confusion_matrix , accuracy_score
cm = confusion_matrix(y_test, y_predd)
ac = accuracy_score(y_test, y_predd)


"Convolution neural network"
"to detect whether dog or cat"
#import libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

#processing the training data
"To avoid overfitting", "Data augementation"
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
training _set = train_datagen.flow_from_directory('dataset/training_set', target_size= (64, 64), batch_size = 32, class_mode = 'binary')

#processing the test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_director('dataset/test_set', target_size=(64, 64),  batch_size=32, class_mode='binary')

#initilizing the CNN
cnn = tf.keras.models.Sequential()

#convilution
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation ='relu', input_shape = [64,64,3]))

#pooling
cnn.add(tf.keras.layers.Maxpooling2D(pool_size = 2, strides = 2)

#add second convultion layer
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation ='relu'))

cnn.add(tf.keras.layers.Maxpooling2D(pool_size = 2, strides = 2))

#flatting
cnn.add(tf.keras.layers.Flatten())

#Full connection
cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))

#outputlayer
cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

#compile CNN
cnn.compile(optimizer= 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#train datatset and evalute test set
"both at the same time"
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

#make single prediction
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size =(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

result = cnn.predict(test_image)
training_set.class_indices

"0,0 = batch and element of batch"
if results[0][0] == 1:
    prediction = 'dog'
    if else:
        prediction = 'cat'
        
        print(prediction)



"Dimesionality reduction"
"principle analysis component"
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_csv('Wine.csv')
x = dataset.iloc[: , :-1].values
y = dataset.iloc[:, -1].values

#split dataset to training and test datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,  y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#apply PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

#logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

#making the cpnfusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)


"linear discriminant analysis LDA"
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_csv('Wine.csv')
x = dataset.iloc[: , :-1].values
y = dataset.iloc[:, -1].values

#split dataset to training and test datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,  y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#apply LDA
"require y_train as it is supervised in contrast pca only require x_test"
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components = 2)
x_train = lda.fit_transform(x_train, y_train)
x_test = lda.transform(x_test)

#logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

#making the cpnfusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)


"kernel PCA"
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_csv('Wine.csv')
x = dataset.iloc[: , :-1].values
y = dataset.iloc[:, -1].values

#split dataset to training and test datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,  y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#apply kernelPCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2, kernel ='rbf')
x_train = kpca.fit_transform(x_train)
x_test = kpca.transform(x_test)

#logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

#making the cpnfusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)


#model selection
"kfold cross validation"
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y), 1 )

#split dataset to training and test datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,  y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)



#training the SVM on the training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train, y_train)

#predict test set
y_pred = classifier.predict(x_test)
np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test),1)), 1)

#making the cpnfusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)

#apply k fold cross validation on TRAINING SET
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier , X = x_train, y = y_train, cv = 10)
print('Acurracy:{:.2f} %'.format(accuracies.mean()*100))
print('standard deviation:{:.2f} %'.format(accuracies.std()*100))


"grid search"
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y), 1 )

#split dataset to training and test datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,  y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)



#training the SVM on the training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train, y_train)

#predict test set
y_pred = classifier.predict(x_test)
np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test),1)), 1)

#making the cpnfusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)

#apply k fold cross validation on TRAINING SET
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier , X = x_train, y = y_train, cv = 10)
print('Acurracy:{:.2f} %'.format(accuracies.mean()*100))
print('standard deviation:{:.2f} %'.format(accuracies.std()*100))

#grid search to find best parameters and therefore best model
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [ 0.5, 0.75, 1], 'kernel':['linear']},
              {'C': [ 0.5, 0.75, 1], 'kernel':['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4]}]
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)
grid_search.fit(x_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print(' Best Acurracy:{:.2f} %'.format(best_accuracy*100))
print('Best parameters:', best_parameters)


"XGBoost"
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y), 1 )

#split dataset to training and test datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,  y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


"""#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)"""

#training the XGBOOST on the training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train, y_train)

#predict test set
y_pred = classifier.predict(x_test)
np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test),1)), 1)

#making the cpnfusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)

#apply k fold cross validation on TRAINING SET
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier , X = x_train, y = y_train, cv = 10)
print('Acurracy:{:.2f} %'.format(accuracies.mean()*100))
print('standard deviation:{:.2f} %'.format(accuracies.std()*100))



"Reinforcment learning"
"upper confidence bound"
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#implement UCB 
import math
N = 10000
d = 10
ads_selected = []
numbers_of_selection = [0] * d
sums_of_rewards = [0] * d
total_Reward = [0]
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if numbers_of_selection[i] > 0:
            average_reward = sums_of_rewards[i] / numbers_of_selection[i]
            delta_i = math.sqrt(3/2* math.log(n + 1) / numbers_of_selection[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
            if(upper_bound > max_upper_bound):
                max_upper_bound = upper_bound
                ad = i
    ads_selected.append(ad)    
    numbers_of_selection[ad] += 1
    sums_of_rewards[ad] = sums_of_rewards[ad] + dataset.values[n, ad]
    total_Reward = total_Reward + dataset.values[n, ad]
    

#visualise the results
plt.hist(ads_selected)
plt.xlabel('ads')
plt.ylabel('number of times each ad was selected')
plt.show()




"thomoson sampling"
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#implement UCB 
import random
N = 10000
d = 10
ads_selected = []
number_of_rewards1 = [0]*d
number_of_rewards0 = [0]*d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(number_of_rewards1[i] + 1, number_of_rewards0[i] + 1)
        if random_beta > max_random:
            ad = i
    ads_selected.append(ad)   
    reward = dataset.values[n, ad]
  if reward == 1:
      number_of_rewards1[ad] = number_of_rewards1[ad] + 1
  else:
      number_of_rewards0[ad] = number_of_rewards0[ad] + 1
      total_reward = total_reward + reward

#visualise the results
plt.hist(ads_selected)
plt.xlabel('ads')
plt.ylabel('number of times each ad was selected')
plt.show()


"Natural language processing"
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Restaurantreviews.tsv', delimiter = '\t', quoting = 3)

#clean the text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
   review = review.lower()
   review = review.split()
   ps = PorterStemmer()
   review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
   review = ' '.join(review)
   corpus.append(review)
  
#create the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CounterVectorizer(max_features = 1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values



#split dataset to training and test datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,  y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

#training the naive bayes on the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

#predict test set
y_pred = classifier.predict(x_test)
np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test),1)), 1)

#making the cpnfusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)






























