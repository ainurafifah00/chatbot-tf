#Libraries for NLP
import nltk
nltk.download('punkt') #for tokenization
from nltk.stem.lancaster import LancasterStemmer #for stemming
stemmer = LancasterStemmer()

#Libraries for Tensorflow processing
import tensorflow as tf
import numpy as np 
import tflearn
import random
import json

with open('intents.json') as json_data:
	intents = json.load(json_data)

#print(intents)

words = []
classes = []
documents = []
ignore = ['?']
#loop through each sentence in the intent's pattern
for intent in intents['intents']:
	for pattern in intent['patterns']:
		#tokenize each word in sentence
		w = nltk.word_tokenize(pattern)
		#print(w)
		#add word to words list
		words.extend(w)
		documents.append((w,intent['tag']))
		if intent['tag'] not in classes:
			classes.append(intent['tag'])

#print(documents)
#print(classes)

#Perform stemming and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore]
words = sorted(list(set(classes)))

#print(words)

#remove duplicate classes
classes = sorted(list(set(classes)))


print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique stemmed words", words)







