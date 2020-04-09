import sys
import heapq
import json
import re
import random

#from coref_resolution import coref
from ne_extraction import ner
from relation_extraction import nre_qg
from dependency_parsing import dp


#Stores wh- and binary patterns for relation-based questions
RE_QG_JSON = open("resources/re_qg.json","r").read()
RE_RULES = json.loads(RE_QG_JSON)

#Stores patterns for dependency-parse based questions
DP_QG_JSON = open("resources/dp_qg.json","r").read()
DP_RULES = json.loads(DP_QG_JSON)

#Controls how far RE goes to detect relationships for each named entity
#Increasing may lead to more relationships at the cost of increased runtime 
RE_GRANULARITY = 5

#What portion of the generated questions should be binary?
BINARY_RATIO = 2/7


class Relationship:
	def __init__(self, _entity1, _entity2, _kind, _score):
		"""
		self.entity1 : Named Entity (typically subject in relation)
		self.entity2 : Named Entity (typicall object in relation)
		self.kind 	 : string (must exist in RE_QG_JSON)
		self.score   : float (must be normalized)
 		"""
		self.entity1 = _entity1 		
		self.entity2 = _entity2  		
		self.kind = _kind       		
		self.score = _score				

	def __eq__(self, other):
		"""
		We define an equivalence class of all relations based on
		relationship's subject entity and its kind
		"""
		return self.kind==other.kind \
		   and (self.entity1.text==other.entity1.text \
			    or self.entity2.text==self.entity2.text)

	def __hash__(self):
		return hash((self.entity1.text,self.kind))




class Question: 
	def __init__(self, text, score):
		"""
		self.text  : string 
			this is the question text
		self.score : float 
			normalized from 0.00 +1.00
		"""
		self.text = text				
		self.score = score 				

	def __lt__(self, other):
		"""
		We will use scores across questions of all kinds 
		for comparison and ranking
		"""
		return (self.score<other.score) 

	def __eq__(self, other):
		return self.text==other.text

	def __hash__(self):
		return hash(self.text)








"""_______________________________RELATION BASED QG________________________________"""


def relationship_to_question(relationship):
	"""
	Takes a discovered relationship and uses the set of rules for relationship 
	based questions in "resources/re_qg.json" to generate a 
	Question along with its score

	INPUT:  Relationship
	OUTPUT: Question
	"""
	if(relationship.kind not in RE_RULES.keys()):
		return Question("No question found.", 0.0)

	pattern = ""
	if(random.randrange(0,10,1)<(BINARY_RATIO*10)):
		pattern = RE_RULES[relationship.kind]["binary"]
	else:
		pattern = RE_RULES[relationship.kind]["pattern"]

	question = pattern.replace('entity1',relationship.entity1.name)
	question = question.replace('entity2',relationship.entity2.name)
	score = relationship.score*RE_RULES[relationship.kind]["quality"]


	if(relationship.entity1.label not in \
		RE_RULES[relationship.kind]["entity1_labels"]):
		score /= 2
	if(relationship.entity2.label not in \
		RE_RULES[relationship.kind]["entity2_labels"]):
		score /= 2
	if(relationship.entity1.name==relationship.entity2.name):
		score /= 2

	return Question(question, round(score,2))



def get_relationships(text,entities):
	"""
	Takes a list of named entities and returns all 
	detected relationships amongst them

	INPUT: List<Named_Entity>
	OUTPUT: List<Relationship>
	"""
	relationships = dict()
	ne_count = len(entities)
	for i in range(ne_count):
		entity1 = entities[i]
		###print("checking "+entity1.name+" for relationships...")

		for j in range(max(0, i-RE_GRANULARITY),min(ne_count,i+RE_GRANULARITY)):
			entity2 = entities[j]
			if((entity1.name,entity2.name) in relationships):
				continue
			(_kind, _score) = nre_qg.infer(text, entity1.start_char \
							, entity1.end_char, entity2.start_char \
							, entity2.end_char)
			relationships[(entity1.name,entity2.name)] = \
						Relationship(entity1, entity2, _kind, _score) 

	return relationships.values()
"""___________________________________________________________________________________"""










"""_______________________________DEPENDENCY BASED QG________________________________"""


def dependency_to_question(entities):
	"""
	CURRIED FUNCTION
	Calling dependency_to_question returns a method that turns dependency-parse
	data into questions only for a specified set of entities
	"""

	def children_to_text(children):
		"""
		Used to turn child objects to clauses from dependency-parse objects
		"""
		n = len(children)
		if(len(children)==0): 
			return ""
		clause = ""
		for i in range(n-1):
			clause += children[i].text + " "
		return clause + children[n-1].text

	def to_question(dependency):
		"""
		dependency_to_question returns this function staged with entities
		"""
		if(None in [dependency.verb, dependency.subject, dependency.object]):
			return Question("No question found",0.0)

		obj_ent = entities[dependency.object.text]
		if(obj_ent.exists):
			full_subject_text = children_to_text(list(dependency.subject.children)) \
							  + dependency.subject.text 
			full_object_text = children_to_text(list(dependency.object.children)) \
							 + dependency.object.text 
			full_verb_text = dependency.verb.lemma_ \
						   + children_to_text(list(dependency.verb.children)[1:])

			pattern = DP_RULES[obj_ent.label]["pattern"]
			quality = DP_RULES[obj_ent.label]["quality"]
			question_text = pattern.replace('SUBJECT',full_subject_text)
			question_text = question_text.replace('OBJECT', full_object_text)
			question_text = question_text.replace('VERB',full_verb_text)
			return Question(question_text,0.5)
		else:
			return Question("No question found.",0.0)
	return to_question


"""___________________________________________________________________________________"""











def main(_filepath, _N):
	f = open(_filepath, "r")
	text = f.read()

	#text = coref.resolve_corefs(text)

	named_entities = ner.extract_ne(text)
	sorted_entities = sorted(list(named_entities.values()),key=ner.get_key)

	relationships = get_relationships(text,sorted_entities)
	relationship_questions = set(map(relationship_to_question, relationships))

	#dependencies = dp.get_dependencies(text)
	#dependency_questions = list(map(dependency_to_question(named_entities),dependencies)) 

	questions = set()
	questions.update(relationship_questions)
	#questions.update(dependency_questions)

	heapq.heapify(list(questions))
	top_questions = heapq.nlargest(int(_N), questions)

	for q in top_questions: print(q.text)

	return












"""____________________________________CLI USAGE______________________________________"""

if __name__ == "__main__":
	argv = sys.argv[1:]
	if(len(argv)!=2):
		print("Usage: <text you want to generate questions from> <number of questions> ")
		sys.exit()
	main(argv[0], argv[1])
"""________________________________________________________________________"""



