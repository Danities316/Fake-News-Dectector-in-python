import numpy as np
import pandas as pd
from itertools import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

#importing your csv file
df = pd.read_csv('news.csv')

df.shape
df.head()

#creating labels
labels = df.label
labels.head()

#spliting your dataset into train and test sets
x_train,x_test,y_train,y_test = train_test_split(df['text'],labels,
                                                test_size=0.2, random_state = 7)
                                                
#DataFlair - Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

DataFlair - Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)
#DataFlair - Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)


print(f"Accuracy: {round(score*100,2)}%")

#DataFlair - Build confusion matrix
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])


#import seaborn as sns
#DataFlair - Build confusion matrix
dada = confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
d = dada[0][0]
d1 = dada[1][1]
d2 = dada[0][1] + dada[1][0]
print("Total No of FAKE NEWS is: ", d)
print("While Total No of Real News is: ", d1)
print("Tota No of wrong prediction is: ", d2)
