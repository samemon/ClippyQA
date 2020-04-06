#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:47:03 2020

@author: Chris
"""

import sys
import random
import re as re

# Attempt to load the article and questions
def load_files():
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
        
# Placeholder method for our actual answering code
def answer_question(question, article_sentences):
    return "I don't know the answer to that, here's a random sentence in the article: " + random.choice(article_sentences)

# Checks that answers are correctly formatted before outputting them
def write_answers(answers):
    for i,answer in enumerate(answers):
        if(answer.count('\n')!=0):
            sys.stderr.write('Answer number ' + str(i) + ' contains a newline, which will mess up formatting\n')
        print(answer)
        

def main():
    (article_text, question_text) = load_files()
    questions = question_text.split('\n')
    
    # A draft way to split articles - still doesn't work on non-ASCII punctuation.
    r = re.compile(r'(\.|\n)')
    article_sentences = r.split(article_text)
    article_sentences = [x for x in article_sentences if len(x)>1]

    answers = [answer_question(q,article_sentences) for q in questions]
    write_answers(answers)
    

if __name__ == "__main__":
    main()
