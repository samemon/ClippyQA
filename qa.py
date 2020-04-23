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
DUMMY_ENTITIES = ['ALEX','GIZMO','MONROEVILLE','1945','AMOUNT']

#Stores patterns for dependency-parse based questions
DP_QG_JSON = open("resources/dp_qg.json","r").read()
DP_RULES = json.loads(DP_QG_JSON)

#Controls how far RE goes to detect relationships for each named entity
#Increasing may lead to more relationships at the cost of increased runtime 
RE_GRANULARITY = 1000





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
		   and (self.entity1.name==other.entity1.name \
			    or self.entity2.name==self.entity2.name)

	def __hash__(self):
		return hash((self.entity1.name,self.kind))

	def __str__(self):
		return "(" + self.entity1.name + ", " + self.entity2.name + ", " \
			       + self.kind + ")"

	def get_entityNames(self):
		return [self.entity1.name, self.entity2.name]






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






def get_relationships(text,local_entities,global_entities):
	"""
	Takes a list of named entities and returns all 
	detected relationships amongst them

	INPUT: String, List<Named_Entity>, List<Named_Entity>
	OUTPUT: Dict<(Named_Entity,Named_Entity), Relationship>
	"""

	relationships = dict()

	for entity1 in local_entities:
		for entity2 in global_entities:
			text_startChar = min(entity1.start_char, entity2.start_char)
			text_endChar = max(entity1.end_char, entity2.end_char)
			text_excerpt = text[text_startChar:text_endChar]
			if(entity1.name==entity2.name):
				continue
			(_kind, _score) = nre_qg.infer(text_excerpt, entity1.start_char \
							, entity1.end_char, entity2.start_char \
							, entity2.end_char)
			relationships[(entity1.name,entity2.name)] = \
						Relationship(entity1, entity2, _kind, _score) 


	return relationships

def answer_questions(article_text, questions):
	"""
	Takes the text from the article as well as list of questions in string form
	and returns a list of answers corresponding to each question

	INPUT: String, List<String>
	OUTPUT: prints answers to STDOUT
	"""

	article_entities = list(ner.extract_ne(article_text).values())
	article_entities = sorted(article_entities,key=ner.get_key)

	for question in questions:

		#Replacing Wh-words with a representative dummy entity to improve OpenNRE performance
		#Also sets grammatically appropriate label(s) for the object of the question
		global_labels = []
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

		if(question==""):
			continue

		if(len(question_entities)==0):
			print("Could not answer question")
			continue

		isBinary = (question[0:2]=="Is" or question[0:3]=="Was" or question[0:3]=="Are")


		

		#find relationships between question_entities in question
		question_relationships = get_relationships(question, question_entities, question_entities)


		if(len(question_relationships)==0):
			print("Could not answer question")
			continue

		#find all instances of the question_entities in the text 
		q_ent_instances = set()
		for q_ent in question_entities:
			if q_ent.name in DUMMY_ENTITIES:
				continue
			startChars = [m.start() for m in re.finditer(q_ent.name, article_text)]
			for startChar in startChars:
				q_ent_instances.add(ner.Named_Entity(q_ent.name, True, startChar, startChar+len(q_ent.name), q_ent.label))

		#examine all relations of question_entities within the text (within a window defined by RE_GRANULARITY)
		best_relationship = Relationship(None,None,"No Kind",0.0)
		for q in q_ent_instances:
			text_startChar = max(q.start_char-RE_GRANULARITY,0)
			text_endChar = min(q.end_char+RE_GRANULARITY,len(article_text))
			text_excerpt = article_text[text_startChar:text_endChar]
			a_ent_instances = ner.extract_ne(text_excerpt).values()

			answer_relationships = get_relationships(article_text, [q], a_ent_instances).values()
			
			for a_rel in answer_relationships:
				is_answer = False
				for q_rel in question_relationships.values():
					is_answer = (q_rel.kind==a_rel.kind and \
								 (a_rel.entity1.name in q_rel.get_entityNames()))
				if is_answer and a_rel.score>best_relationship.score:
					best_relationship = a_rel
				if best_relationship.score==0.0 and a_rel.entity2.label in global_labels:
					best_relationship = a_rel
					best_relationship.score = 0.01

		#Answers if a relevant relationship can be found
		if(best_relationship.score>0.0):
			if(isBinary):
				print("Yes")
			else:
				print(best_relationship.entity2.name.replace("\n",""))
		else:
			if(isBinary):
				print("No")
			else:
				print("Could not answer question")





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