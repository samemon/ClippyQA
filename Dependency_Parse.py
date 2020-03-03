import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("He gave him the apple.")
qword = ""

# Get dependencies of each token
dependencies = {}
for token in doc:
	if token.dep_ == "ROOT":
		dependencies[token.dep_] = token.lemma_
	else:
		dependencies[token.dep_] = token.text.lower()

# See if any Named Entities in Document, adjust question words based on it
if len(doc.ents) != 0:
	for ent in doc.ents:
		if ent.label_ == 'GPE' or ent.label_ == 'ORG':
			qword = "Where"
		elif ent.label_ == 'MONEY':
			qword = "How much"
else:
	if "prep" in dependencies.keys():
		qword = "Where"
	else:
		qword = "What"

# Rule-based question generation based on whether "is" is in sentence or if there are prepositions

if qword != "":
	if "is" not in doc.text:
		print(qword + " does " + dependencies['nsubj'] + " " + dependencies['ROOT'] + "?")
	elif "prep" in dependencies.keys() and "pobj" in dependencies.keys():
		print("Is " + dependencies['nsubj'] + " " + dependencies['ROOT'] + " " + dependencies['prep'] 
			+ " " + dependencies['pobj'] + "?")
	else:
		print(qword + " is " + dependencies['nsubj'] + "?")



# To improve:
# What to do about names that are tagged as "ORG" but should be names of real people
# How to distinguish prepositions of location (He went to the store) vs prepositions of relationships
# (He gave the apple to him.) 
# What to do about different tenses: Make everything present or adjust based on tense? 

