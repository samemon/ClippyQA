import sys
import heapq
import json
import re

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import gensim

from ne_extraction import ner
from relation_extraction import nre_qg
from dependency_parsing import dp

from gensim import corpora, models, similarities

#Stores similarities between relation types
RELATION_CLASSES = open("resources/equivalent_labels.json","r").read()
EQ_CLASSES = json.loads(RELATION_CLASSES)

#Stores wh- and binary patterns for relation-based questions
RE_QG_JSON = open("resources/re_qg.json","r").read()
RE_RULES = json.loads(RE_QG_JSON)

RELATION_KINDS = ['PERSON','NORP','FAC','ORG','GPE','LOC','PRODUCT','EVENT','WORK_OF_ART','LAW',\
				  'LANGUAGE','DATE','TIME','PERCENT','MONEY','QUANITY','ORDINAL','CARDINAL']

DUMMY_ENTITIES = ['ALEX','GIZMO','MONROEVILLE','1945','AMOUNT']

AUXILLARY_VERBS = ['Are', 'Is', 'Was', 'Were', 'Being', 'Been', 'Can', 'Could', 'Do' 'Does', 'Did', 'Have', 'Has', 'Had', 'Will', 'Would.',\
				   'are', 'is', 'was', 'were', 'being', 'been', 'can', 'could', 'do' 'does', 'did', 'have', 'has', 'had', 'will', 'would.']

#Stores patterns for dependency-parse based questions
DP_QG_JSON = open("resources/dp_qg.json","r").read()
DP_RULES = json.loads(DP_QG_JSON)

#Controls how far RE goes to detect relationships for each named entity
#Increasing may lead to more relationships at the cost of increased runtime 
RE_GRANULARITY = 20





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





def return_similar(article_text, question, N):
    file_docs = []

    tokens = sent_tokenize(article_text)
    for line in tokens:
        file_docs.append(line)

    gen_docs = [[w.lower() for w in word_tokenize(text)] 
            for text in file_docs]

    dictionary = gensim.corpora.Dictionary(gen_docs)
    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
    
    tf_idf = gensim.models.TfidfModel(corpus)

    sims = gensim.similarities.Similarity('workdir_sims/',tf_idf[corpus],
                                        num_features=len(dictionary))

    query_doc = [w.lower() for w in word_tokenize(question)]
    query_doc_bow = dictionary.doc2bow(query_doc)
    
    
    query_doc_tf_idf = tf_idf[query_doc_bow]
    # print(document_number, document_similarity)
    similarity = sims[query_doc_tf_idf]
    top_N = similarity.argsort()[-N:][::-1]

    results = []
    for i in top_N:
    	results.append(file_docs[i])

    return results

    #print('Comparing Result:', sims[query_doc_tf_idf])



#memoize discovered relationships 
RELATION_DICT = dict()


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
			if (entity1.name,entity2.name) in RELATION_DICT.keys():
				relationships[(entity1.name,entity2.name)] = RELATION_DICT[(entity1.name,entity2.name)]
				continue

			text_startChar = min(entity1.start_char, entity2.start_char)
			text_endChar = max(entity1.end_char, entity2.end_char)
			text_excerpt = text[text_startChar:text_endChar]
			if(entity1.name==entity2.name):
				continue

			(_kind, _score) = nre_qg.infer(text, entity1.start_char \
							, entity1.end_char, entity2.start_char \
							, entity2.end_char)

			if _kind in RE_RULES.keys():
				if(entity1.label not in RE_RULES[_kind]["entity1_labels"]):
					_score = 0.0
				if(entity2.label not in RE_RULES[_kind]["entity2_labels"]):
					_score = 0.0
			else:
				_score = 0.0

			relationships[(entity1.name,entity2.name)] = \
						Relationship(entity1, entity2, _kind, _score) 

			RELATION_DICT[(entity1.name,entity2.name)] = relationships[(entity1.name,entity2.name)]



			#print(relationships[(entity1.name,entity2.name)])


	return relationships

def replace_dummy_entities(question):
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

	for wh_word in ["how many",'How many']:
		if wh_word in question.split():
			question = question.replace(wh_word,'AMOUNT')
			global_labels = ['PERCENT','MONEY','QUANITY','CARDINAL']

		question = question.replace("'s","")
		return (question, global_labels)




def answer_questions_with_nre(article_text, questions):
	"""
	Takes the text from the article as well as list of questions in string form
	and returns a list of answers corresponding to each question

	INPUT: String, List<String>
	OUTPUT: prints answers to STDOUT
	"""
	for question in questions:

		isBinary = question.split(" ")[0] in AUXILLARY_VERBS

		#get relevant texts to search from
		passages = return_similar(article_text,question,RE_GRANULARITY)

		#replace wh- words with dummy enitities
		(question, global_labels) = replace_dummy_entities(question)

		#extract named entities and determine the subject
		question_entities = list(ner.extract_ne(question).values())

		#ignore blank questions
		if(question==""):
			continue

		
		#find relationships between question_entities in question
		question_relationships = get_relationships(question, question_entities, question_entities).values()
		question_entities = [i for i in question_entities if i.name not in DUMMY_ENTITIES] 

		if(len(passages)==0): #no passages found similar to question
			print("Could not answer question")
			continue
		elif(len(question_entities)==0): #no subject found in question
			print(passages[0])
			continue
		elif(len(question_relationships)==0): #no topic found in question
			print(passages[0])
			continue
		


		#get kinds of relationships found in the questions
		question_relationship_kinds = []
		for q_rel in question_relationships:
			question_relationship_kinds.append(q_rel.kind)

		#get similar relationships to ones above
		similar_relationship_kinds = []
		for kind in question_relationship_kinds:
			if kind in EQ_CLASSES.keys():
				similar_relationship_kinds += EQ_CLASSES[kind]


		best_relationship = Relationship(None,None,"No Kind",0.0)
		guess_relationship = Relationship(None,None,"No Kind",0.0)
		best_passage = passages[0]
		for passage in passages:
			#print(passage)
			passage_entities = list(ner.extract_ne(passage).values())
			passage_relationships = get_relationships(passage, passage_entities+question_entities, passage_entities).values()
			for p_rel in passage_relationships:
				if(p_rel.entity2 in question_entities):
					continue
				if p_rel.kind in question_relationship_kinds and p_rel.score>best_relationship.score:
					best_relationship = p_rel
				if p_rel.kind in similar_relationship_kinds and p_rel.score>guess_relationship.score:
					guess_relationship = p_rel
				if p_rel.kind in global_labels:
					best_passage = passage



		#Answers if a relevant relationship can be found
		if(best_relationship.score>0.0):
			if(isBinary):
				print("Yes")
			else:
				print(best_relationship.entity2.name.replace("\n",""))
		elif(guess_relationship.score>0.0):
			if(isBinary):
				print("Yes",guess_relationship.score)
			else:
				print(guess_relationship.entity2.name.replace("\n",""))
		else:
			if(isBinary):
				print("No")
			else:
				print(best_passage)






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

    answers = answer_questions_with_nre(article_text,questions)
    #write_answers(answers)
    



if __name__ == "__main__":
    main()