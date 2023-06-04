import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Read the data 
df = pd.read_csv('/Users/anujkrishnapandit/Desktop/news.csv')

# Get shape and head
df.head()
df.shape

# DataFlair - Get the labels
labels = df.label
labels.head()

# DataFlair - Split the dataset
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

# DataFlair - Initialize a TfidfVectorizer
tfidf_vectorized = TfidfVectorizer(stop_words='english', max_df=0.7)

# DataFlair - Fit and transform train set, transform test set
t_train = tfidf_vectorized.fit_transform(x_train)
t_test = tfidf_vectorized.transform(x_test)

# DataFlair - Initialize a PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(t_train, y_train)

# DataFlair - Predict on the test set and calculate accuracy
y_pared = pac.predict(t_test)
score = accuracy_score(y_test, y_pared)
print(f'Accuracy: {round(score*100,2)}%')

# DataFlair - Build confusion matrix
confusion_matrix(y_test, y_pared, labels=['FAKE', 'REAL'])

