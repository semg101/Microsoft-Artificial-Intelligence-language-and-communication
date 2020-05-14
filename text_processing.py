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


#Read a document and print its content-----------------------------------------------------------------------------------------1
doc1 = open("process.txt", "r")
doc1Txt = doc1.read()
print(doc1Txt)


#Normalize the text-------------------------------------------------------------------------------------------------------------2
from string import punctuation

  #remove the numeric digits
txt = ''.join(c for c in doc1Txt if not c.isdigit())
  # remove the punctuation and make the lower case
txt = ''.join(c for c in txt if c not in punctuation).lower()
  #print the normalized text
print(txt)



#Analyse the frequency of each word----------------------------------------------------------------------------------------------3
import nltk
import pandas as pd
from nltk.probability import FreqDist
nltk.download("punkt")
 
 #Tokenize the text into individual words
words1 = nltk.tokenize.word_tokenize(txt)
 #Get the frequency distribution of the words into a data frame 
fdist = FreqDist(words1)
count_frame = pd.DataFrame(fdist, index = [0]).T
count_frame.columns = ['Count']
print(count_frame)


#Plot the distribution as a pareto chart------------------------------------------------------------------------------------------4
import matplotlib.pyplot as plt

 #Sort the data frame by frequency
counts = count_frame.sort_values('Count', ascending = False)
 #Display the top 60 words as a bar plot
fig = plt.figure(figsize=(16, 9))
ax = fig.gca()
counts['Count'][:60].plot(kind = 'bar', ax = ax)
ax.set_title('Frequency of the most common words')
ax.set_ylabel('Frequency of word')
ax.set_xlabel('word')
#plt.show()


#Remove the stop words--------------------------------------------------------------------------------------------------------------5
 #Get the standard stop words from NLTK
nltk.download("stopwords")
from nltk.corpus import stopwords
 #Filter out the stop words 
txt = '   '.join([word for word in txt.split() if word not in (stopwords.words('english'))])
print(txt)
 #Get the frequency distribution of the remaining words
words2 = nltk.tokenize.word_tokenize(txt)
fdist = FreqDist(words2)
count_frame = pd.DataFrame(fdist, index = [0]).T
count_frame.columns = ['Count']
 #Plot the frequency of the top 60 words
counts = count_frame.sort_values('Count', ascending = False)
fig = plt.figure(figsize = (16, 9))
ax = fig.gca()
counts['Count'][:60].plot(kind = 'bar', ax = ax)
ax.set_title('Frequency of the most common words')
ax.set_ylabel('Frequency of word')
ax.set_xlabel('word')
plt.show()