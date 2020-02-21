import spacy
import en_core_web_sm
import time

nlp = en_core_web_sm.load()


import neuralcoref
neuralcoref.add_to_pipe(nlp)

print("Input a string - include he/she/etc for coref to resolve")
text = input()


#Some sample strings to try

#Working examples:

#text='My sister has a dog. She loves him.'    #Detects sister=she, dog=him
#text='The President today announced a new government program. It would promote things worth promoting, he said.'
#^ Ties The President to He and 'a new government program' to 'it'. 

#Kinda working:

#text='Jim is a rich man. He shaves daily with a razor. It is gold-plated.' #Detects Jim=He, but not Razor=It
#text='Johnny was so adjective, even Sarah had to admit Johhny was more adjective than she was.' #Detects Johnny=he, but says 'even Sarah'=she.

doc = nlp(text)

print("String as entered: ")
print(text)

#Example coref replacement. Some weirdness needed to go from spacy tokens to python strings.
text_revised=text
for i in range(len(doc._.coref_clusters)):             #Coref clusters is a list of cluster objects
    cluster = doc._.coref_clusters[i]
    print('Coref cluster ' + str(i) + ':')
    words_in_cluster=[]
    for l in cluster.mentions:                         #Each cluster objects contains mentions, a list of lists of tokens e.g. [[Jon,Snow],[he]]
        tokens_combined=""
        for text in l:
            tokens_combined+=str(text) + ' '
        words_in_cluster.append(tokens_combined.strip())
    print(words_in_cluster)
    text_revised = text_revised.replace(words_in_cluster[1],words_in_cluster[0])

print("Revised Text: ")
print(text_revised)