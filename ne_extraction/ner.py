import spacy
from collections import defaultdict

nlp = spacy.load("en_core_web_sm")

class Named_Entity:
	"""
	All named entities extracted from tools like spaCy, NLTK, etc.
	should be encoded as objects of this class. Named_Entity supports
	strings of all kinds, even ones not in the given document!

	name : string 
		The text of this object, such as "John", "Apple Inc.", etc.
	exists : boolean
		Whether or not the named entity actually exists in the doc
	self.start_char : int
		The start position of the name in the doc. This is -1 for 
		Named_Entity objects that are not in the doc
	self.end_char : int
		The end position. Also -1 when not in the doc
	self.label : string
		The entity type, see https://spacy.io/api/annotation
		for the kinds of labels available under OntoNotes 5. 
	"""
	def __init__(self, _name, _exists, _start_char, _end_char, _label):
		self.name = _name
		self.exists = _exists
		self.start_char = _start_char
		self.end_char = _end_char
		self.label = _label

	def __str__(self):
		return self.name


def extract_ne(text):
	"""
	This function will return a dictionary
	mapping strings to the named entity objects
	they represent, if any. 

	Args:
		text: string 
			should be pre-processed with coref resolution for best results

	Returns: 
		named_entity_dict : Dict<string, Named_Entity>
			if a string S is a named entity within the document, then
			named_entity_dict[S] is the Named_Entity object for S
	"""
	
	doc = nlp(text)
	named_entity_dict = defaultdict(lambda: Named_Entity("Invalid Entity", False, -1, -1, ""))

	for ent in doc.ents:
		named_entity_dict[ent.text] = Named_Entity(ent.text, True, ent.start_char, ent.end_char, ent.label_)

	return named_entity_dict

def get_key(named_entity):
	return named_entity.start_char



