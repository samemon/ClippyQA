# ClippyQA
This repository is an implementation of a Question Generation (QG) and Question Answering system.

# Question Generation (QG)
Given an article/text, the system generates questions based on *relationships* between entities. The system diagram for the question generation system is below:

[]()

The question generation happens in two phases: (i) Using Named-Entity recognition (NRE-QG), and (ii) Using constituency parsing (PARSE-QG). The questions generated are then weighted and ranked using several criteria. The system diagrams for each of the two question generators are as follows:

## NRE-QG
[]()

## PARSE-QG

[]()

## Question Ranker

We conduct question ranking in two phases: (i) pre-generation ranking, and (ii) post-generation ranking

### Pre-generation ranking

Pre-generation ranking is applied to NRE-QG system as follows:

- Using relationship confidence score between entities as a proxy for question quality
- Deducting quality score for questions with incorrect entity labels, and self-relations

Pre-generation ranking is applied to NRE-PARSE system as follows:

- Using TextRank to extract only important sentences from the passage to generate important questions

### Post-generation ranking

Post-generation ranking is applied as follows:

- Discarding questions with less than 5 tokens to avoid meaningless questions
- Discarding questions with length greater than 30 to be concise
- Weight “WH”, and “binary” questions to diversify the selection of questions generated
- Weight the two QG systems to diversify the selection of questions generated

# Question Answering (QA)
Given an article/text and a question, the system answers the question based on the text.

QA system is designed in the following steps:

- Preprocessing the text: Tokenization and Removal of stop words
- Use Gensim to retrieve relevant passages
- Extract relations from passages using OpenNRE as shown in the diagram below
- Find closest relation to question
- Ranking System

[]()

# Novelty
In terms of the novelty of the system, our system is novel in the following ways:
- Using NRE as a main question generation engine
- Using TextRank to generate important questions
- Generating “NOT” questions
- Diversifying our questions using constituency parsing
- Ranking questions using both pre- and post-generation approaches

# Tools

- OpenNRE (https://github.com/thunlp/OpenNRE): Used for relationship extraction for the QG phase
- NLTK (https://www.nltk.org/): Used for preprocessing
- SpaCy (https://spacy.io/): Used for named-entity recognition and preprocessing
- StanfordCoreNLP (https://github.com/stanfordnlp/CoreNLP): Used for constituency parsing in the QG phrase
- Gensim (https://pypi.org/project/gensim/): Used for text summarization (using TextRank algorithm) and for sentence similarity
- NeuralCoref + BERT (https://github.com/huggingface/neuralcoref): Used for co-reference resolution in QG and sentence similarity in QA


# Further Details

The project is further explained in detail in the video here: https://www.youtube.com/watch?v=1V2_XTjff2c

# Contributers

- Shahan Ali Memon
- Rigved Deshpande
- Christopher Bradsher
- Lazar Andjelic

