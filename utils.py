import pickle
import codecs
import re

def generate_ngram(input_list, n):
    result = []
    for i in range(1, n+1):
        result.extend(zip(*[input_list[j:] for j in range(i)]))
    return result

def gets_voc_set(path):
    with codecs.open(path,'r','utf-8') as f: 
        voc_set = [line.strip() for line in f]
    return set(voc_set)

def load_data(path):
    with codecs.open(path,'r','utf-8') as f: 
        all_corp = [re.sub(' +', ' ',str(line).lower().replace(')',' ').replace('(',' ').replace('（',' ').replace('）',' ').strip()) for line in f]
    return all_corp

def score_formulation(PMI, entropy, probability):
    if PMI==None:
        return None
    return (PMI+entropy)*probability
    
def save_model(model, filename):
    with open(filename, 'wb') as fw:
        pickle.dump(model, fw)

def load_model(filename):
    with open(filename, 'rb') as fr:
        model = pickle.load(fr)
    return model
