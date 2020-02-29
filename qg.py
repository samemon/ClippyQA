import sys
import heapq

from coref_resolution import coref
from ne_extraction import ner
from relation_extraction import nre_qg

class Relationship:
	def __init__(self, e1, e2, r):
		self.entity1 = e1
		self.entity2 = e2
		self.relation = r

class Named_Entity:
	def __init__(self, text, start_char, end_char, _label):
		self.text = text
		self.start_char = start_char
		self.end_char = end_char
		self._label = _label

class Question:
	def __init__(self, text, score):
		self.text = text
		self.score = score

	def __lt__(self, other):
		return (self.score<other.score)

def relation_qg(entity1, entity2, relation):

	kind = relation[0]
	score = relation[1]
	question = ""

	if(entity1.text==entity2.text):
		score = 0.01

	if(kind == 'residence'):
		if(entity1._label!='PERSON' or (entity2._label not in ['LOC','GPE'])):
			score = 0.01
		question ="Question: Where does "+entity1.text+" live?: "+entity2.text

	elif(kind == 'work location'):
		if(entity1._label!='PERSON' or (entity2._label not in ['ORG','GPE','LOC','FAC'])):
			score = 0.01
		question = "Question: Where does "+entity1.text+ " work?: "+ entity2.text

	elif(kind == 'occupation'):
		if(entity1._label!='PERSON'):
			score = 0.01
		question = "Question: What does "+ entity1.text + " do?: "+entity2.text

	elif(kind == 'sibling'):
		if(entity1._label!='PERSON' or entity2._label!='PERSON'):
			score = 0.01
		question = "Question: Who is "+entity1.text+ " related to?: "+entity2.text

	elif(kind == 'headquarters location'):
		print(entity1._label, entity2._label)
		if(entity1._label not in ['LOC','FAC','ORG'] or entity2._label not in ['GPE','LOC']):
			score = 0.01
		question = "Question: Where is "+entity1.text+ " located?: "+entity2.text

	elif(kind == 'publisher'):
		question = "Question: Who is "+entity1.text+ "'s publisher?: "+entity2.text

	elif(kind == 'notable work'):
		question = "What is a notable work of "+entity1.text+ "?: "+entity2.text

	elif(kind == 'has part'):
		if(entity2._label=='PERSON'):
			score = 0.01
		question = "Question: " + entity2.text + " had a part in which work?: " + entity1.text

	elif(kind == 'characters'):
		score = 0.01
		question = "What were " + entity2.text + " and " + entity1.text + "?: characters"

	elif(kind == 'owned by'):
		if(entity1._label not in ['GPE','FAC','ORG']):
			score = 0.01
		question = "Question: Who is "+entity1.text+ " owned by?: "+entity2.text

	elif(kind == 'follows'):
		score = 0.01
		question = "Question: Who or what does " + entity2.text + " follow?: "+entity1.text

	elif(kind == 'followed by'):
		score = 0.01
		question = "Question: Who or what does " + entity1.text + " follow?: "+entity2.text

	elif(kind == 'part of'):
		question = "Question: What is " + entity1.text + " part of?: "+entity2.text

	else:
		question = "Could not generate question for " + kind + ". "

	return Question(question, score)

def main(_filepath, _N):
	f = open(_filepath, "r")
	text = f.read()

	text = coref.resolve_corefs(text)
	entities = ner.extract_ne(text)
	ne_count = len(entities)
	print(str(ne_count)+" named entities detected")

	relationship_data = []
	for i in range(ne_count):
		print("checking "+entities[i].text+" for relationships...")
		for j in range(i, ne_count):
			if(i==j):
				continue
			entity1 = Named_Entity(entities[i].text, entities[i].start_char, entities[i].end_char, entities[i].label_)
			entity2 = Named_Entity(entities[j].text, entities[j].start_char, entities[j].end_char, entities[j].label_)

			relation = nre_qg.infer(text, entity1.start_char, entity1.end_char, entity2.start_char, entity2.end_char)
			relationship_data.append( Relationship(entity1, entity2, relation) )

	relation_questions =[]
	for rd in relationship_data:
		rd_question = relation_qg(rd.entity1, rd.entity2, rd.relation)
		relation_questions.append(rd_question)


	questions = []
	questions += relation_questions
	#TODO: Dependency parsing score

	print(str(len(questions))+" questions generated.")

	heapq.heapify(questions)
	top_questions = heapq.nlargest(int(_N), questions)

	for q in top_questions:
		print(q.text, q.score)

	return

if __name__ == "__main__":
	argv = sys.argv[1:]
	if(len(argv) != 2):
		print("Usage: <text you want to generate questions from> <number of questions> ")
		sys.exit()

	main(argv[0], argv[1])