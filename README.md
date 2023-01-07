# Iris-Flower-Classification:

Steps to Classify Iris Flower:

1. Load the data
2. Analyze and visualize the dataset
3. Model training.
4. Model Evaluation.
5. Testing the model.

Step 1 – Load the data:

# DataFlair Iris Flower Classification
# Import Packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
%matplotlib inline

Numpy will be used for any computational operations.
We’ll use Matplotlib and seaborn for data visualization.
Pandas help to load data from various sources like local storage, database, excel file, CSV file, etc.

# Load the data
df = pd.read_csv('iris.data', names=columns)
df.head()


df.head() only shows the first 5 rows from the data set table.
data head


Step 2 – Analyze and visualize the dataset:

Let’s see some information about the dataset.

# Some basic statistical analysis about the data
df.describe()
describe

From this description, we can see all the descriptions about the data, like average length and width, minimum value, maximum value, the 25%, 50%, and 75% distribution value, etc.

Let’s visualize the dataset.

# Visualize the whole dataset
sns.pairplot(df, hue='Class_labels')

To visualize the whole dataset we used the seaborn pair plot method. It plots the whole dataset’s information.
visualize

From this visualization, we can tell that iris-setosa is well separated from the other two flowers.
And iris virginica is the longest flower and iris setosa is the shortest.
Now let’s plot the average of each feature of each class.

# Here we separated the features from the target value.


Step 3 – Model training:

# Split the data to train and test dataset.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
Using train_test_split we split the whole data into training and testing datasets. Later we’ll use the testing dataset to check the accuracy of the model.

# Support vector machine algorithm
from sklearn.svm import SVC
svn = SVC()
svn.fit(X_train, y_train)

Here we imported a support vector classifier from the scikit-learn support vector machine.
Then, we created an object and named it svn.
After that, we feed the training dataset into the algorithm by using the svn.fit() method.

Step 4 – Model Evaluation:

# Predict from the test dataset
predictions = svn.predict(X_test)

# Calculate the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)
Now we predict the classes from the test dataset using our trained model.
Then we check the accuracy score of the predicted classes.
accuracy_score() takes true values and predicted values and returns the percentage of accuracy.
Output:
0.9666666666666667

The accuracy is above 96%.


Step 5 – Testing the model:

X_new = np.array([[3, 2, 1, 0.2], [  4.9, 2.2, 3.8, 1.1 ], [  5.3, 2.5, 4.6, 1.9 ]])
#Prediction of the species from the input vector
prediction = svn.predict(X_new)
print("Prediction of Species: {}".format(prediction))
Here we take some random values based on the average plot to see if the model can predict accurately.
Output:

Prediction of Species: [‘Iris-setosa’ ‘Iris-versicolor’ ‘Iris-virginica’]



