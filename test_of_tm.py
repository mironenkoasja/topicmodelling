# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 16:12:38 2019

@author: Asja
"""

# To Do
# 1 Написать код для подгрузки 4 баз И слияния их в одну
# 2 Код для тематического моделирования
# 3 Код для выгрузки баз

#dataset types
#1 - sociology
#2 - cardiology
#3 - literature

# columns in dataset
# 1 - language
# 2 - keywords
# 3 - keywords_2
# 4 - abstract
# 5 - Citations rate
# 6 - WOS_id
# 7 - TYPE


import os
os.chdir('C:/Users/Asja/Documents/Python Scripts/lingdan/topic_modelling/education_2017')
import string

import spacy
nlp = spacy.load('en')

import csv
import numpy as np
import pandas as pd
import re, nltk, spacy, gensim
from nltk import FreqDist
from nltk.corpus import stopwords

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import Word2Vec

# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
from pyLDAvis import gensim as gensim_pyLDAvis
%matplotlib inline



# Import and preprocessing Dataset
## Functions
def read_csv_list(path):
    result_list = []
    with open(path, encoding='utf-8') as obj:            
        reader = csv.reader(obj)
        for row in reader:
            result_list.append(row)        
    return result_list

table = str.maketrans({key: None for key in string.punctuation})

def only_one_column(info_list, i):
    result_list = []
    for item in info_list:
        result_list.append(item[i].translate(table).lower())
    return result_list    



# Merge all sets
l_paths = ['sociology_2017.csv', 'cardio_2017.csv', 'literature_2017.csv']
all_sets = []
for path in l_paths:
    dataset = read_csv_list(path) 
    all_sets += dataset[1:len(dataset)]
len(all_sets)

# Only abstracts
abstr_all_sets = only_one_column(all_sets, 3)


# Identification of most frequent words to add to stop-words
new_str = ''
for i in abstr_all_sets:
    new_str += i

# Most frequent Words
nltk.download('punkt')

ns_tok = nltk.word_tokenize(new_str)

fdist = FreqDist(ns_tok)   
fdist.most_common(20)


# LDA with gensim

# Tokenize each sentence into a list of words, removing punctuations 
# and unnecessary characters altogether
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(abstr_all_sets))

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=4, threshold=50) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=50)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
# print(trigram_mod[bigram_mod[data_words[0]]])

# remove stop_words, make bigrams, trigrams
stop_words = stopwords.words('english')
stop_words.extend([])
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

#print(data_lemmatized[:1])

    
# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
# print(corpus[13:14])

# Build LDA model
lda_model_g = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=6, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=200,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

pprint(lda_model_g.print_topics())
doc_lda = lda_model_g[corpus]

#len(doc_lda)
#
#print(doc_lda[0])
#
##TODOTODOTODOTODO  3 Код для выгрузки баз
#
#
## Compute Perplexity
#print('\nPerplexity: ', lda_model_g.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
#
## Compute Coherence Score
#coherence_model_lda = CoherenceModel(model=lda_model_gl, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
#coherence_lda = coherence_model_lda.get_coherence()
#print('\nCoherence Score: ', coherence_lda)
#
#
#
#def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
#    """
#    Compute c_v coherence for various number of topics
#
#    Parameters:
#    ----------
#    dictionary : Gensim dictionary
#    corpus : Gensim corpus
#    texts : List of input texts
#    limit : Max num of topics
#
#    Returns:
#    -------
#    model_list : List of LDA topic models
#    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
#    """
#    coherence_values = []
#    model_list = []
#    for num_topics in range(start, limit, step):
#        model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word)
#        model_list.append(model)
#        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
#        coherence_values.append(coherencemodel.get_coherence())
#
#    return model_list, coherence_values
#
#
#model_list, coherence_values = compute_coherence_values(dictionary=id2word, 
#                                                        corpus=corpus, 
#                                                        texts=texts, 
#                                                        start=2, limit=50, step=4)
#
## Show graph
#limit=50; start=2; step=4;
#x = range(start, limit, step)
#plt.plot(x, coherence_values)
#plt.xticks(np.arange(min(x), max(x)+1, 4))
#plt.xlabel("Num Topics")
#plt.ylabel("Coherence score")
#plt.legend(("coherence_values"), loc='best')
#plt.show()
#
#outfile = open('sets.csv', 'w', newline='')
#writer = csv.writer(outfile)
#for i in all_sets:    
#    writer.writerow(i)
#outfile.close()
#
#lda_model_g[corpus][1]
#Finding the dominant topic in each document

def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model_g, corpus=corpus, texts=abstr_all_sets)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.head(10)
df_dominant_topic.to_csv('out2.csv')


lda_model_g[corpus][0]

mixture = [dict(lda_model_g[x]) for x in corpus]
pd.DataFrame(mixture).to_csv("topic_mixture.csv")


# Visualize the topics
pyLDAvis.enable_notebook()
pyLDAvis.disable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model_g, corpus, id2word)
pyLDAvis.show(vis)
pyLDAvis.save_html(vis, '18_literature.html')

