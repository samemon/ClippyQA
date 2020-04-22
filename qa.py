import sys
import heapq
import json
import re

import nltk

#from coref_resolution import coref
from ne_extraction import ner
from relation_extraction import nre_qg
from dependency_parsing import dp


#Stores wh- and binary patterns for relation-based questions
RE_QG_JSON = open("resources/re_qg.json","r").read()
RE_RULES = json.loads(RE_QG_JSON)
RELATION_KINDS = ['PERSON','NORP','FAC','ORG','GPE','LOC','PRODUCT','EVENT','WORK_OF_ART','LAW',\
				  'LANGUAGE','DATE','TIME','PERCENT','MONEY','QUANITY','ORDINAL','CARDINAL']

#Stores patterns for dependency-parse based questions
DP_QG_JSON = open("resources/dp_qg.json","r").read()
DP_RULES = json.loads(DP_QG_JSON)

#Controls how far RE goes to detect relationships for each named entity
#Increasing may lead to more relationships at the cost of increased runtime 
RE_GRANULARITY = 100





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

	def __str__(self):
		return "(" + self.entity1.name + ", " + self.entity2.name + ", " \
			       + self.kind + ")"






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






def get_relationships(text,local_entities,global_entities, local_labels, global_labels):
	"""
	Takes a list of named entities and returns all 
	detected relationships amongst them

	INPUT: String, List<Named_Entity>, List<Named_Entity>
	OUTPUT: Dict<(Named_Entity,Named_Entity), Relationship>
	"""

	relationships = dict()

	for entity1 in local_entities:
		if(entity1.label not in local_labels):
			continue
		for entity2 in global_entities:
			if(entity2.label not in global_labels):
				continue
			(_kind, _score) = nre_qg.infer(text, entity1.start_char \
							, entity1.end_char, entity2.start_char \
							, entity2.end_char)
			relationships[(entity1.name,entity2.name)] = \
						Relationship(entity1, entity2, _kind, _score) 


	return relationships



def get_relation_info(relationships):
	"""
	A dictionary of dictionaries of named entities!
	The data structure returned by this is used as follows

	info[relation.kind] returns a dictionary of all relationships 
						that are of kind relation.kind

	info[relation.kind][subject_entity] returns a list of all entities that share
										a relationship of kind relation.kind with
										subject_entity

	INPUT: Dict<(Named_Entity, Named_Entity),Relationship>
	OUTPUT: Dict<  
				Relationship.Kind,  
				Dict< Named_Entity.name,
					  List<Named_Entity.name>  
					 >   
				>

	"""
	info = dict()
	for relation in relationships.values():
		if(relation.kind not in info.keys()):
			info[relation.kind] = [relation]
		else:
			info[relation.kind].append(relation)

	for kind in info.keys():
		temp = dict()
		for relation in info[kind]:
			if(relation.entity1==relation.entity2):
				continue
			if(relation.entity1 not in temp):
				temp[relation.entity1.name] = [relation.entity2.name]
			else:
				temp[relation.entity1.name].append(relation.entity2.name)
		info[kind] = temp

	return info



def answer_questions(article_text, questions):
	"""
	Takes the text from the article as well as list of questions in string form
	and returns a list of answers corresponding to each question

	INPUT: String, List<String>
	OUTPUT: prints answers to STDOUT
	"""

	#Build up knowledge base from document
	article_entities = list(ner.extract_ne(article_text).values())
	article_entities = sorted(article_entities,key=ner.get_key)

	answers = []
	for question in questions:

		global_labels = []
		local_labels = []

		#Replacing Wh-words with a representative dummy entity to improve OpenNRE performance
		#Also sets grammatically appropriate label(s) for the object of the question
		for wh_word in ["who",'Who']:
			if wh_word in question.split():
				question = question.replace(wh_word,'ALEX')
				global_labels = ['PERSON']

		for wh_word in ["what",'What']:
			if wh_word in question.split():
				question = question.replace(wh_word,'GIZMO')
				global_labels = ['PRODUCT','WORK_OF_ART','LAW']

		for wh_word in ["where",'Where']:
			if wh_word in question.split():
				question = question.replace(wh_word,'MONROEVILLE')
				global_labels = ['FAC','LOC','GPE']

		for wh_word in ["when",'When']:
			if wh_word in question.split():
				question = question.replace(wh_word,'1945')
				global_labels = ['TIME','DATE','EVENT']

		for wh_word in ["whom",'Whom']:
			if wh_word in question.split():
				question = question.replace(wh_word,'ALEX')
				global_labels = ['PERSON','ORG']
			
		for wh_word in ["how much",'How much']:
			if wh_word in question.split():
				question = question.replace(wh_word,'AMOUNT')
				global_labels = ['PERCENT','MONEY','QUANITY','CARDINAL']

		question = question.replace("'s","")

		#extract named entities and determine the subject
		question_entities = list(ner.extract_ne(question).values())
		question_entities = sorted(question_entities,key=ner.get_key)

		#Determine grammatically appropriate label(s) for the subject of the question
		for question_entity in question_entities:
			if question_entity.name in ['ALEX','GIZMO','MONROEVILLE','1945','AMOUNT']:
				continue
			else:
				local_labels = [question_entity.label]

		if(question==""):
			continue

		if(len(question_entities)==0):
			print("Could not answer question. (Subject unclear)")
			continue

		#find relationships between question_entities in question
		question_relationships = get_relationships(question, question_entities, question_entities, RELATION_KINDS, RELATION_KINDS)
		if(len(question_relationships)==0):
			print("Could not answer question. (Relation unclear)")
			continue

		#find relationships between entities in question_text and article_text
		article_relationships = get_relationships(article_text, question_entities+article_entities, article_entities, local_labels, global_labels)
		relation_info = get_relation_info(article_relationships)


		#try to find the relationships in question to the knowledge base
		answer_is_found = False
		answer=""
		for topic in question_relationships.values():
			if(answer_is_found):
				break
			if(topic.kind in relation_info.keys()):
				if(topic.entity1.name in relation_info[topic.kind].keys()):
					answer = relation_info[topic.kind][topic.entity1.name][0]
					answer_is_found = True
				elif(topic.entity2.name in relation_info[topic.kind].keys()):
					answer = relation_info[topic.kind][topic.entity2.name][0]
					answer_is_found = True
		if(answer_is_found):
			print(answer)
		else:

			print("Could not answer question. (Answer not found)")







def load_files():
	"""
	Attempt to load the article and questions

	INPUT: None
	OUTPUT: String, String
	"""

	try:
		article_location = sys.argv[1]
		question_location = sys.argv[2]
	except IndexError:
		raise IndexError("Not enough arguments provided")
	try:
		with open(article_location,'r',encoding='utf-8') as f:
			article_contents = f.read()
	except OSError:
		raise OSError("File "+article_location + " not found. Code was run from " + sys.argv[0])
	try:
		with open(question_location,'r',encoding='utf-8') as f:
			question_contents = f.read()
	except OSError:
		raise OSError("File "+question_location + " not found. Code was run from " + sys.argv[0])
	return (article_contents,question_contents)





def main():
    (article_text, question_text) = load_files()
    questions = question_text.split('\n')
    
    # A draft way to split articles - still doesn't work on non-ASCII punctuation.
    r = re.compile(r'(\.|\n)')

    answers = answer_questions(article_text,questions)
    #write_answers(answers)
    



if __name__ == "__main__":
    main()