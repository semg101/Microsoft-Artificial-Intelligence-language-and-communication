import sys
import numpy
import pandas
import matplotlib
import seaborn
import scipy
import sklearn

print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(numpy.__version__))
print('Pandas: {}'.format(pandas.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Seaborn: {}'.format(seaborn.__version__))
print('Sklearn: {}'.format(sklearn.__version__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


#Read the first document and print its content-----------------------------------------------------------------------------------------1
doc1 = open("process.txt", "r")
doc1Txt = doc1.read()
print(doc1Txt)

#Normalize the text
from string import punctuation
  #remove the numeric digits
txt = ''.join(c for c in doc1Txt if not c.isdigit())
  #remove the punctuation and make the lower case
txt = ''.join(c for c in txt if c not in punctuation).lower()
  #Remove the stop words
import nltk
  #Get the standard stop words from NLTK
nltk.download("stopwords")
from nltk.corpus import stopwords
txt = '   '.join([word for word in txt.split() if word not in (stopwords.words('english'))])

print("-----------------------------------------------------------------------")
#Get the second document, normalize it and remove the stops words-----------------------------------------------------------------------2
 #Read a document and print its content
doc2 = open("process2.txt", "r")
doc2Txt = doc2.read()
print(doc2Txt)

 #Normalize the text
from string import punctuation
  #remove the numeric digits
txt2 = ''.join(c for c in doc2Txt if not c.isdigit())
  #remove the punctuation and make the lower case
txt2 = ''.join(c for c in txt2 if c not in punctuation).lower()
  #Remove the stop words
import nltk
  #Get the standard stop words from NLTK
nltk.download("stopwords")
from nltk.corpus import stopwords
txt2 = '   '.join([word for word in txt2.split() if word not in (stopwords.words('english'))])


print("-----------------------------------------------------------------------")
#Get the third document, normalize it and remove the stops words-------------------------------------------------------------------------3
 #Read a document and print its content
doc3 = open("process3.txt", "r")
doc3Txt = doc3.read()
print(doc2Txt)

 #Normalize the text
from string import punctuation
  #remove the numeric digits
txt3 = ''.join(c for c in doc3Txt if not c.isdigit())
  # remove the punctuation and make the lower case
txt3 = ''.join(c for c in txt3 if c not in punctuation).lower()
  #Remove the stop words
import nltk
  #Get the standard stop words from NLTK
nltk.download("stopwords")
from nltk.corpus import stopwords
txt3 = '   '.join([word for word in txt3.split() if word not in (stopwords.words('english'))])



#Get TF-IDF values from the top three words in each document---------------------------------------------------------4
 #install textblob library and define functions for TF-IDF
#pip install -U textblob
import math
from textblob import TextBlob as tb

#Term frequency
def tf(word, doc):
	return doc.words.count(word)/len(doc.words)

#The occurrence of a word in the list of documents
def contains(word, docs):
	return sum(1 for doc in docs if word in doc.words)

#Inverse document frequency
def idf(word, docs):
	return math.log(len(docs)/(1 + contains(word, docs)))

def tfidf(word, doc, docs):
	return tf(word, doc) * idf(word, docs)

#Create a collection of documents as textblobs
doc1 = tb(txt)
doc2 = tb(txt2)
doc3 = tb(txt3)
docs = [doc1, doc2, doc3]

#Use TF-IDF to get the three most important words from each document
print("-----------------------------------------------------------------------")
for i, doc in enumerate(docs):
	print("Top words in document {}".format(i + 1))
	scores = {word: tfidf(word, doc, docs) for word in doc.words}
	sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
	for word, score in sorted_words[:3]:
		print ("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))