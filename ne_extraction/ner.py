import spacy
nlp = spacy.load("en_core_web_sm")

def extract_ne(text):
	"""
	This function will return a list 
	of (start_index, end_index) tuples
	corresponding to every named entity
	detected within the input text

	Args:
		text: |string| should be pre-processed with coref resolution for best results
	Returns: 
		ne_indices: entity list containing (text, start_char, end_char, label_)
	"""
	
	doc = nlp(text)
	return doc.ents

