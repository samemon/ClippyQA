# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:13:17 2020

@author: Chris
"""
import sys
from sklearn.metrics.pairwise import cosine_similarity
import re
import time
from sentence_transformers import SentenceTransformer



def answer_questions(article_sentences, questions):
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    
    t0 = time.time()
    article_embeddings = model.encode(article_sentences)
    #print("Article encode time: ")
    #print(time.time()-t0)
    answers = []
    for q in questions:
        question_embedding = model.encode([q])[0]
        best_answer_index, best_answer_score = -1,-1
        for (i,a) in enumerate(article_embeddings):
            #print(article_sentences[i])
            cos_sim = cosine_similarity([a],[question_embedding])
            #print(cos_sim)
            if cos_sim>best_answer_score:
                best_answer_score = cos_sim
                best_answer_index = i
        answers.append(article_sentences[best_answer_index])
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
    
    
    #https://stackoverflow.com/questions/25735644/python-regex-for-splitting-text-into-sentences-sentence-tokenizing
    r = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
    article_lines = article_text.split('\n')
    article_sentences = []
    for l in article_lines:
        article_sentences.append(r.split(l))
    article_sentences = [item for sublist in article_sentences for item in sublist]
    while("" in article_sentences):
        article_sentences.remove("")
        
    
    #print("Article: ")
    #print(article_sentences)
    #print("Questions:")
    #print(questions)
    

    answers = answer_questions(article_sentences,questions)
    write_answers(answers)
    



if __name__ == "__main__":
    main()
