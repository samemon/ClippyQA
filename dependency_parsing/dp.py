import spacy
import sys
import itertools
import functools

nlp = spacy.load("en_core_web_sm")

class Dependency:
	def __init__(self):
		self.subject = None
		self.verb = None
		self.object = None

def flatten_tokens(words):
	sentence = ""
	for word in words:
		sentence += " " + word.text
	return sentence


def get_dependency_from_sentence(sentence):
	doc = nlp(sentence)
	dependency = Dependency()

	for token in doc:
		if(token.dep_=="nsubj"):
			dependency.subject = token

		elif(token.dep_=="ROOT"):
			dependency.verb = token

		elif(token.dep_ in ["pobj","dobj"]):
			dependency.object = token

	return dependency




def get_dependencies(text):
	return list(map(get_dependency_from_sentence,text.split(". ")))




if __name__ == "__main__":
	argv = sys.argv[1:]
	if(len(argv) != 1):
		print("Usage: <text you want to parse from> ")
		sys.exit()

	print(get_dependencies(argv[0]))
	
