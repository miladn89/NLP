# NLP
NLP and text mining on COVID-19 experts Twitter accounts

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 03:01:27 2020

@author: Admin
"""
import torch, transformers
import re
import pandas as pd 
import numpy as np 
import torch as to 
from tokenizers import ByteLevelBPETokenizer, BertWordPieceTokenizer, CharBPETokenizer, SentencePieceBPETokenizer
from tokenizers import pre_tokenizers
from transformers import *
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers import pre_tokenizers
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import json
import string
import nltk
from nltk.stem.porter import *
import os
from sklearn.utils import shuffle
import requests
import pickle
import ast
import glob
import os
path = r'C:\Users\Owner\OneDrive - Mississippi State University\Desktop\twitter nlp\old_data'
files2 = []
for i in os.listdir(path):
    if os.path.splitext(i)[1] == '.csv':
        files2.append(i)
files2 = files2[::-1]        
import torch
print('Torch', torch.__version__, 'CUDA', torch.version.cuda)
print('Device:', torch.device('cuda:0'))
device = torch.device('cuda:0')   
# item = 0
# def my_func(item):
for item in range(len(files2)):    
    print(files2[item])
    device = torch.device('cpu')        
    whole  = pd.read_csv(os.path.join(path, files2[item]), index_col=0)
    for i , j in enumerate(whole.full_text):
        temp = re.sub(r"'", '', j)
        temp = re.sub(r'"', '', temp)
        temp = re.sub(r'https?:\/\/.*', '', temp)
        temp = re.sub(r"\{", '', temp)
        temp = re.sub(r"\}", '', temp)
        temp = re.sub(r"\[", '', temp)
        temp = re.sub(r"\]", '', temp)
        temp = re.sub(r"\(", '', temp)
        temp = re.sub(r"\)", '', temp)   
        temp = re.sub(r"\\n", '', temp)
        temp = re.sub(r"\â", '', temp)
        temp = re.sub(r"\â", '', temp)
        temp = re.sub(r"\â", '', temp)      
        whole.full_text[i] = temp
    target = pd.DataFrame(index=whole.index) 
    target['id'] = whole.id
    # # whole = shuffle(whole)
    # # whole = shuffle(whole)
    # # split_index = int(whole.shape[0]/15)
    # # test = whole.iloc[:split_index,:]
    # # train = whole.iloc[split_index+1:,:]
    # PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
    # tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    # print('dataframe')
    # # sample_txt = 'When was I last outside? I am stuck at home for 2 weeks.'
    # def encoder(txt):
    #     return tokenizer.encode_plus(
    #       txt,
    #       max_length=32,
    #       add_special_tokens=True, # Add '[CLS]' and '[SEP]'
    #       return_token_type_ids=False,
    #       padding='longest',
    #       truncation=True,
    #       return_attention_mask=True,
    #       return_tensors='pt',  # Return PyTorch tensors
    #     ).to(device)
    # encoding = [[] for i in range(whole.shape[0])]
    # tokens = [[] for i in range(whole.shape[0])]
    # for i, j in enumerate(whole.full_text):
    #     tokens[i] = tokenizer.tokenize(j)
    #     encoding[i] = encoder(j)
    # target['encoding'] = encoding
    # target['tokens'] = tokens 

    # print('encoding')
    # # Sentiment analysis pipeline
    # sentiment_model =  pipeline('sentiment-analysis')
    # # # Question answering pipeline, specifying the checkpoint identifier
    # # pipeline('question-answering', model='distilbert-base-cased-distilled-squad', tokenizer='bert-base-cased')
    # # # Named entity recognition pipeline, passing in a specific model and tokenizer
    # # model = AutoModelForTokenClassification.from_prewholeed("dbmdz/bert-large-cased-finetuned-conll03-english")
    # # tokenizer = AutoTokenizer.from_prewholeed("bert-base-cased")
    # # pipeline('ner', model=model, tokenizer=tokenizer)

    # target_sentiment = [[] for i in range(whole.shape[0])]
    # for i,j in enumerate(tokens):
    #     # print(i,j)
    #     try: 
    #     # print("======")
    #         target_sentiment[i] = sentiment_model(j)
    #     except: 
    #         target_sentiment[i] = {'label': 'POSITIVE', 'score': 0.55}
    #         print(f'sentiment;instance: {i};{files2[item]}')

    # target['sentiments'] =  target_sentiment
    # print('sentiment')
    summarizer_model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer2 = T5Tokenizer.from_pretrained('t5-small')
    target_summarizer = [[] for i in range(whole.shape[0])]
    for i,j in enumerate(whole['full_text']):
        if i % 500 == 0:    
            print(i)
        try : 
            preprocess_text = j.strip().replace("\n","")
            t5_prepared_Text = "summarize: "+preprocess_text
            tokenized_text = tokenizer2.encode(t5_prepared_Text, return_tensors="pt").to(device)
            temp = summarizer_model.generate(tokenized_text,
                                            num_beams=5,
                                            no_repeat_ngram_size=2,
                                            min_length=30,
                                            max_length=100,
                                            early_stopping=True)
            target_summarizer[i] = tokenizer2.decode(temp[0], skip_special_tokens=True)
        except: 
            print(f'summarization;instance: {i}; {files2[item]}')
    target['summariztion'] = target_summarizer
    print('summarization')
#     namedEntityRecgnition = pipeline("ner")
#     target_NER = [[] for i in range(whole.shape[0])]
#     for i,j in enumerate(whole['full_text']):
#         try: 
#             target_NER[i] = namedEntityRecgnition(j)
#         except: 
#             print(f'summarization;instance: {i}; {files2[item]}')
#     target['NER'] = target_NER
#     print('NER')
    target.to_csv(fr'C:\Users\Owner\OneDrive - Mississippi State University\Desktop\twitter nlp\output\{files2[item]}')
    print('='*50)
    
#     # return target

# # for i in range(len(files2)):
#   # temp = my_func(i)
#   # df = pd.Series(temp,index=temp.keys())
#   # df.to_csv(f'E:\twitter nlp\{files2[i]}')


# =============================================================================
#
item = 0
for item in range(len(files2)):    
    print(files2[item])
    device = torch.device('cpu')        
    whole  = pd.read_csv(os.path.join(path, files2[item]), index_col=0) 
    target = pd.DataFrame(index=whole.index) 
    target['id'] = whole.id
    target['full_text'] = whole.full_text
    hashtag = [[] for i in range(len(whole))]
    user_mention = [[] for i in range(len(whole))]
    for i, j in enumerate(whole.entities):
        temp = ast.literal_eval(j)
        try: 
            if len(temp['hashtags']) >=1: 
                hashtag[i] = [temp['hashtags'][i]['text'] for i in range(len(temp['hashtags']))]
            else:
                hashtag[i] = temp['hashtags']
            if len(temp['user_mentions']) >=1: 
                user_mention[i] = [temp['user_mentions'][i]['name'] for i in range(len(temp['user_mentions']))]
            else:
                user_mention[i] = temp['user_mentions']
        except:
            print(i)
    target['hashtags'] = hashtag
    target['user_mentions'] = user_mention      
    target.to_csv(fr'E:\twitter nlp\new_data_hashtags\{files2[item][:-4]}_hashtags.csv')
    print('='*50)
    
    
    
    
# =============================================================================
path = r'E:\twitter nlp\new_data_hashtags'
files3 = []
for i in os.listdir(path):
    if os.path.splitext(i)[1] == '.csv':
        files3.append(i)
all_hashtags= [[] for i in range(len(files3))]
for item in range(len(files3)): 
    print(files3[item])    
    whole = pd.read_csv(fr'E:\twitter nlp\new_data_hashtags\{files3[item]}')
    for i in whole.hashtags:
        if len(i) >= 1:
            all_hashtags[item].extend(ast.literal_eval(i))
all_unique_hashtags = [[] for i in range(len(files3))]
for i, item in enumerate(all_hashtags):
    all_unique_hashtags[i] = list(set(item))
    

unique_all_unique_hashtags = []
for i in all_unique_hashtags:
    unique_all_unique_hashtags.extend(i)









# =============================================================================
# 
import pkg_resources
from symspellpy import SymSpell, Verbosity

sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
if sym_spell.word_count:
    pass
else:
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
# =============================================================================

# =============================================================================
# topic modeling 
import sys
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import re
import string
import gensim
from gensim.utils import simple_preprocess
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from gensim.models import CoherenceModel
import spacy
import pyLDAvis
import pyLDAvis.gensim  
import matplotlib.pyplot as plt
import time
import datetime

path = r'E:\twitter nlp\new_data_hashtags'
files3 = []
for i in os.listdir(path):
    if os.path.splitext(i)[1] == '.csv':
        files3.append(i)
data = [[] for i in range(len(files3))]        
for item in range(len(files3)):
    print(files3[item])    
    whole = pd.read_csv(fr'E:\twitter nlp\new_data_hashtags\{files3[item]}')
    for i , j in enumerate(whole.full_text):
        temp = re.sub(r"'", '', j)
        temp = re.sub(r'"', '', temp)
        temp = re.sub(r'https?:\/\/.*', '', temp)
        temp = re.sub(r"\{", '', temp)
        temp = re.sub(r"\}", '', temp)
        temp = re.sub(r"\[", '', temp)
        temp = re.sub(r"\]", '', temp)
        temp = re.sub(r"\(", '', temp)
        temp = re.sub(r"\)", '', temp)   
        temp = re.sub(r"\\n", '', temp)
        temp = re.sub(r"\â", '', temp)
        temp = re.sub(r"\â", '', temp)
        temp = re.sub(r"\â", '', temp) 
        whole.full_text[i] = temp
    data[item] = ' '.join(whole.full_text)
    

data = [re.sub(r'\s', ' ', sent) for sent in data]
#removing numbers
data = [re.sub(r'\d', '', sent) for sent in data]

#removing punks
for j in string.punctuation:
    data = [re.sub(r"\{}".format(j), '', sent)for sent in data]
# Remove new line characters
data = [re.sub(r'\s+', ' ', sent) for sent in data]
data = [w.lower() for w in data]


from gensim.parsing.preprocessing import STOPWORDS
swgensim = set(STOPWORDS)
from nltk.corpus import stopwords
swnltk = set(stopwords.words('english'))
import spacy
import spacy.lang.en
swspacy = set(spacy.lang.en.stop_words.STOP_WORDS)
mystopword = swspacy | swnltk | swgensim



# =============================================================================
stemmer = SnowballStemmer("english")
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in mystopword and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

word_tokenized = [preprocess(sent) for sent in data]


# Build the bigram and trigram models
bigram = gensim.models.Phrases(word_tokenized) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[word_tokenized])  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

def make_bigramst(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigramst(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

# Form Bigrams
data_words_bigrams = make_bigramst(word_tokenized)
mywords = make_trigramst(data_words_bigrams)

#####################################################
mywords= all_unique_hashtags
#####################################################
# Create Dictionary
id2word = gensim.corpora.Dictionary(mywords)
# Create Corpus
texts = mywords
corpus = [id2word.doc2bow(text) for text in texts]
# =============================================================================
tfidf_model= [f'tfidfmodel{i}' for i in range(3 ,30, 3)]
for i in range(len(tfidf_model)):
  tfidf_model[i] = []
j=0
for i in range(3, 30, 3):
    Num_Topics = i
    tfidf_model[j] = gensim.models.tfidfmodel.TfidfModel(corpus=corpus,
                                           id2word=id2word,
                                           normalize=True)
   
    j+=1

ldamc_tfidf_model_new = [f'ldamctfidfmodelnew{i}' for i in range(3 , 30, 3)]
for i in range(len(ldamc_tfidf_model_new)):
    ldamc_tfidf_model_new[i] = []
  
j=0
for i in range(3, 30 ,3):
    Num_Topics = i
    ldamc_tfidf_model_new[j] = gensim.models.ldamulticore.LdaMulticore(
                                    corpus=tfidf_model[j][corpus],
                                    id2word=id2word,
                                    num_topics=Num_Topics, 
                                    eval_every = 1,
                                    workers=5,
                                    per_word_topics=True)
  
    j+=1
    print(j)
# =============================================================================

# =============================================================================
ldamctfidfdict20 = {}
ldamctfidflist20 = []

for num in range(len(ldamc_tfidf_model_new)):    
     count=0  
     for topic in ldamc_tfidf_model_new[num].print_topics(num, num_words=30):
         count+=1
         print(topic)
         print("="*50)
         ldamctfidflist20.append(topic)
     ldamctfidfdict20[f'ldamctfidfmodel#{num}']= ldamctfidflist20
     ldamctfidflist20= []
     print(num)

ldamctfidfperplexity= {}
ldamctfidfcoherence= {}
ldamctfidftimelist = []
ldamctfidfscore={}
j=0
for num in range(len(ldamc_tfidf_model_new)):
    start = time.time()
#    ldamctfidfscore[f'ldamctfidfscore#{num}'] = ldamc_tfidf_model_new[num].score(tfidf_model[j][corpus])
#    print("Log Likelihood: ", ldamctfidfscore[f'ldamctfidfscore#{num}'])
      # a measure of how good the model is. lower the better.
    ldamctfidfperplexity[f'ldamctfidfperplexity#{num}'] = ldamc_tfidf_model_new[num].log_perplexity(tfidf_model[j][corpus])
    print(f'\nPerplexitymodel#{num}: ', ldamctfidfperplexity[f'ldamctfidfperplexity#{num}'])
                  # Compute Coherence Score
    coherence_model_ldamctfidf = CoherenceModel(model=ldamc_tfidf_model_new[num], 
                                         texts=texts, 
                                         dictionary=id2word, 
                                         coherence='c_v')
    ldamctfidfcoherence[f'coherencemodel#{num}'] = coherence_model_ldamctfidf.get_coherence()
    print(f'\nCoherence Score#{num}: ', ldamctfidfcoherence[f'coherencemodel#{num}'])
    end = time.time()
    ldamctfidftimelist.append(end - start)
    print(end - start)
    j+=1

# =============================================================================

# =============================================================================
# Visualize the topics
ldamctfidfvis = {}
timelistpyldavisldamctfidf = [] 
for num in range(len(ldamc_tfidf_model_new)-1,-1, -1):
#    num = 
    start = time.time()
#    pyLDAvis.enable_notebook()
    ldamctfidfvis[f"ldamctfidfvismodel#{num}"] = pyLDAvis.gensim.prepare(ldamc_tfidf_model_new[num], 
           corpus,
           id2word, 
           sort_topics=False,
           mds='mmds')
#    pyLDAvis.display(ldavis[f"ldavismodel#{num}"] )
    pyLDAvis.save_html(ldamctfidfvis[f"ldamctfidfvismodel#{num}"], f'ldamctfidfhtmlvis{num}.html')
    end = time.time()
    timelistpyldavisldamctfidf.append(end - start)
    print(f'processing time is: {end - start}')
    
    
    
    
