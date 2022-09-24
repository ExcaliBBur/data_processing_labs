import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import * 
import string
import random


#randomize 2 genres
lst = ['Christian', 'Country', 'Pop', 'Rock', 'R&B'] 
genre1 = random.choice(lst)
genre2 = random.choice(lst)
while (genre1 == genre2):
    genre2 = random.choice(lst)
print("Your genres is", genre1, "and", genre2)
 
nltk.download('omw-1.4', quiet=True)
data = pd.read_csv("lab2\dataset.csv")
columns = data[['genre','lyrics']]
columns = columns[(columns.genre == genre1) | (columns.genre == genre2)]

#to lower case
def toLowerCase():
    lowered = columns['lyrics'].str.lower()
    columns['lowered'] = lowered
    
toLowerCase()

#delete punctuation
def deletePunctuation():
    withoutPunctuation = columns['lowered'].apply(lambda x: [char for char in x if char not in string.punctuation])
    without_punctuation = []
    for a in withoutPunctuation:
        without_punctuation.append(''.join(a))   
    columns['without_punctuation'] = without_punctuation

deletePunctuation()

#tokenize
def tokenize():
    token = columns.apply(lambda row: nltk.word_tokenize(row['without_punctuation']), axis=1)
    columns['tokened'] = token
    
nltk.download('punkt', quiet=True)
tokenize()

#delete stop words
def deleteStopWords():
    noise = stopwords.words('english')
    withoutstop = columns['tokened'].apply(lambda x: [item for item in x if item not in noise])
    without_stop = []
    for a in withoutstop:    
        without_stop.append(", ".join(a))
    columns['without_stop'] = without_stop

nltk.download('stopwords', quiet=True)
deleteStopWords()

#lemmatization
def lemmatization():
    lemmatizer = WordNetLemmatizer()
    lemmatized = columns['without_stop'].apply(lambda x: [lemmatizer.lemmatize(x)])
    lemma = []
    for a in lemmatized:    
        lemma.append(", ".join(a))
    columns['lemmatized'] = lemma
    
nltk.download('wordnet', quiet=True)
lemmatization()

#train and test data 
x_train, x_test, y_train, y_test = train_test_split(columns.lemmatized, columns.genre, train_size = 0.7)

#vectorization 
vectorizer = CountVectorizer(ngram_range=(1, 3))
vectorized_x_train = vectorizer.fit_transform(x_train)

#classification
clf = MultinomialNB()
clf.fit(vectorized_x_train, y_train)

del columns

# task 2 (find song myself)
#my genres and songs
data = pd.read_csv("lab2/mydata.csv")
columns = data[['genre','lyrics']]
columns = columns[(columns.genre == genre1) | (columns.genre == genre2)]

toLowerCase()
deletePunctuation()
tokenize()
deleteStopWords()
lemmatization()

x_test = columns['lemmatized']
y_test = columns['genre']
vectorized_y_test = vectorizer.transform(y_test)
vectorized_x_test = vectorizer.transform(x_test)
pred = clf.predict(vectorized_x_test)
print("Conclusion about your genres:", pred.flat[0], "and", pred.flat[1])
print(classification_report(y_test, pred, zero_division = 0))
print("Total accuracy is", np.mean(pred == y_test)*100,"%")

input("\nEnter smth to continue\n")

# task 3 (difference between david-bowie and paul-mccartney)
data = pd.read_csv("lab2/dataset-lyrics-musics-mini.csv")
columns = data[['genre','lyrics']]
columns = columns[(columns.genre == 'david-bowie') | (columns.genre == 'paul-mccartney')]

toLowerCase()
deletePunctuation()
tokenize()
deleteStopWords()
lemmatization()

x_train, x_test, y_train, y_test = train_test_split(columns.lemmatized, columns.genre, train_size = 0.7)

vectorizer = CountVectorizer(ngram_range=(1, 3))
vectorized_x_train = vectorizer.fit_transform(x_train)

clf = MultinomialNB()
clf.fit(vectorized_x_train, y_train)
vectorized_x_test = vectorizer.transform(x_test)
pred = clf.predict(vectorized_x_test)
print(classification_report(y_test ,pred))
print("Accuracy is", np.mean(pred == y_test)*100 , "%")