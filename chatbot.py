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


#BUILDING MODEL
#create training data
training = []
output = []

#create an empty array for output
output_empty = [0] * len(classes)

#create training set, bag of words
for doc in documents:
	bag = []
	#list of tokenized words for the pattern
	pattern_words = doc[0] 
	#stemming each word
	pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
	#create bag of words array
	for w in words:
		bag.append(1) if w in pattern_words else bag.append(0)

	output_row = list(output_empty)
	output_row[classes.index(doc[1])] = 1

	training.append([bag, output_row])


#shuffling features and turning it it into np.array
random.shuffle(training)
training = np.array(training)


#print(training)

#creating training lists
train_x = list(training[:,0])
train_y = list(training[:,1])

print("X", train_x)
print("Y", train_y)


#resetting underlying graph data
tf.compat.v1.reset_default_graph()

#Building neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

#defining model and setting up tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

#Start training
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')




