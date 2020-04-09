import sys
import heapq
import json
import re

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






def get_relationships(text,entities):
	"""
	Takes a list of named entities and returns all 
	detected relationships amongst them

	INPUT: List<Named_Entity>
	OUTPUT: Dict<(Named_Entity,Named_Entity), Relationship>
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
				Dict< Named_Entity,
					  List<Named_Entity>  
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
				temp[relation.entity1] = [relation.entity2]
			else:
				temp[relation.entity1].append(relation.entity2)
		info[kind] = temp

	return info



def answer_questions(article_text, questions):
	"""
	Takes the text from the article as well as list of questions in string form
	and returns a list of answers corresponding to each question

	INPUT: String, List<String>
	OUTPUT: List<String>
	"""

	###article_text = coref.resolve_corefs(article_text)

	#Build up knowledge base from document
	named_entities = ner.extract_ne(article_text)
	relationships = get_relationships(article_text,list(named_entities.values()))
	relation_info = get_relation_info(relationships)

	answers = []
	for question in questions:
		if(question==""):
			continue

		#extract named entities and determine the subject
		question_entities = list(ner.extract_ne(question).values())
		if(len(question_entities)==0):
			answers.append("Could not answer question. (Subject unclear)")
			continue

		#pick out highest accuracy relationship (likely the topic of the question)
		question_relationships = get_relationships(question,question_entities)
		if(len(question_relationships)==0):
			answers.append("Could not answer question. (Relation unclear)")
			continue

		#find most accurate relationships (this is likely the question topic)
		topic = Relationship(None,None,"NO RELATION",0.0)
		for relationship in question_relationships.values():
			if(relationship.score >= topic.score):
				topic = relationship
		if(topic.score==0.0):
			answers.append("Could not answer question. (Topic unclear)")
			continue

		#match topic relationship to existing dictionary of relationships
		if(topic.kind in relation_info.keys()):
			knowledge_base = relation_info[topic.kind]
			if(topic.entity1 in knowledge_base.keys()):
				answer_entity = knowledge_base[topic.entity1][0]
				answers.append(answer_entity.name)
			elif(topic.entity2 in knowledge_base.keys()):
				answer_entity = knowledge_base[topic.entity2][0]
				answers.append(answer_entity.name)
			else:
				backup_answer = list(knowledge_base.values())[0]
				answers.append(backup_answer[0].name)
		else:
			backup_answer = list(list(knowledge_base.values())[0])[0]
			answers.append(backup_answer.name)

	return answers






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



def write_answers(answers):
	"""
	Checks that answers are correctly formatted before outputting them

	Input: List<String>
	Output: STDOUT
	"""
	for i,answer in enumerate(answers):
		if(answer.count('\n')!=0):
			sys.stderr.write('Answer number ' + str(i) + ' contains a newline, which will mess up formatting\n')
		print(answer)





def main():
    (article_text, question_text) = load_files()
    questions = question_text.split('\n')
    
    # A draft way to split articles - still doesn't work on non-ASCII punctuation.
    r = re.compile(r'(\.|\n)')

    answers = answer_questions(article_text,questions)
    write_answers(answers)
    



if __name__ == "__main__":
    main()