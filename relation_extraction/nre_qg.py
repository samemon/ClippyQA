'''
Description: This program is a simple 
wrapper to generate questions based on the
task of relationship extraction using openNRE

Author: Shahan Ali Memon
'''

__author__      = "Shahan A. Memon"
__copyright__   = "Copyright 2020, Carnegie Mellon"


import sys
import opennre
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
import spacy
import itertools
from itertools import combinations

MODEL_NRE = 'wiki80_bert_softmax'


def infer(model,text,pos_one_st,pos_one_end,
		pos_two_st,pos_two_end):

		"""
		This function will be used to infer 
		the relationship between two entities 
		in text[pos_one_st:post_one_end] and 
		text[pos_two_st:pos_two_end]

		Args:
			model: the model for openNRE
			text: Any line of text
			pos_one_st: start index of first entitity
			pos_one_end: end index of first entity
			pos_two_st: start index of second entity
			pos_two_end: end index of second entity
		Returns: 
			type of relationship (string) e.g. residence, or
			headquarters location, etc.
		"""
		return model.infer({'text': text, 
				'h': {'pos': (pos_one_st, pos_one_end)}, 
				't': {'pos': (pos_two_st, pos_two_end)}})

def get_indices(text):
	"""
	This function takes in text string and returns 
	the start and end index of each word entity.

	Args:
		text: A normalized preprocessed proper string 
		of a sentence
	Returns:
		[(start,end),(start,end),....]
	"""

	'''
	cachedStopWords = stopwords.words("english")
	text = ' '.join([word for word in text.split() 
			if word not in cachedStopWords])

	'''

	span_generator = WhitespaceTokenizer().span_tokenize(text)
	spans = [span for span in span_generator]
	return spans

def create_question(entity1, entity2, relationship):
	"""
	This function will take in two entities and 
	create a rule based question based on the inferred 
	relationship between them
	Args:
		- entity1: string 
		- entity2: string
		- relationship: one of the 80 relationships
	"""

	if(relationship == 'residence'):
		return "Question: Where does "+entity1+" live?: "+entity2

	elif(relationship == 'work location'):
		return  "Question: Where does "+entity1+ " work?: "+entity2

	elif(relationship ==  'occupation'):
		return "Question: What does "+entity1+ " do?: "+entity2

	else:
		return "Question Rule not specified for the relationship"



if __name__ == "__main__":
	argv = sys.argv[1:]
	if(len(argv) != 1):
		print("Usage: <text you want to extract from>")
		sys.exit()

	text = argv[0]
	# Let us load the NRE model first
	model = opennre.get_model(MODEL_NRE)
	indices = get_indices(text)
	tokens = text.split()
	cachedStopWords = stopwords.words("english")
	all_pairs = [list(map(tuple, comb)) for comb in combinations(indices, 2)]

	# Let us now get all the relations
	for i, j in all_pairs:
		if(i != j):
			token1 = text[i[0]:i[1]]
			token2 = text[j[0]:j[1]]
			if(not(token1 in cachedStopWords) and \
					not(token2 in cachedStopWords)):
				r = infer(model,text,i[0],i[1],j[0],j[1])
				if(r[1] > 0.8):
					print("relation between (",token1,") and (",\
							token2, ") is: ", r[0], " with prob ",
							r[1])

					print("##############")
					print("##############")
					print(create_question(token1, token2, r[0]))
					print("##############")
					print("##############")

