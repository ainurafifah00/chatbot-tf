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

