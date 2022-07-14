import math
from collections import Counter
import tqdm
from commons.phrases_mining.utils import *

class Node:
    """
    Build node of trie tree
    """
    def __init__(self, char):
        self.char = char
        self.word_finish = False
        self.count = 0
        self.child = {}
        self.child.clear()
        self.count_back = 0
        self.word_finish_back = False
        self.PMI = None
        self.entropy = None
        self.probability = None

class Trietree(object):
    """
    Build Trie tree to record left, right words and their Frequece.
    """
    def __init__(self, node):
        """
        Initial first layer tree.
        Parm node: root node char.
        parm data: Input token frequency dictionary.
        """
        self.root = Node(node)
        
    def search(self, words):
        '''
        Parm phrase: String with one or more token which splitted by white space.
        Output: Tuple of frequency, PMI, entropy and probability of given phrase.
        '''
        if isinstance(words, str):
            words = words.split(' ')
        elif not isinstance(words, tuple) and not isinstance(words, list):
            raise ValueError("Not support words as "+str(type(words)))
        node = self.root
        for i in words:
            if node:
                node = node.child.get(i)
        if node:
            return (node.count, node.PMI, node.entropy, node.probability)
        return (None, None, None, None, None)
                
    def add(self, word, freq):
        # a->b->c path
        length = len(word)
        node = self.root
        for count, char in enumerate(word):
            # find the node path in tree or build node
            if node.child.get(char)!=None:
                node = node.child.get(char)
            else:
                new_node = Node(char)
                node.child[char]=new_node
                node = new_node
            # check whether it is a complete word or not
            if count == length-1:
                node.count+= freq
                node.word_finish = True
        # b->c->a path (to build the left entropy)
        node = self.root
        if length > 1:
            word = list(word)
            word=word[1::]+[word[0]]
            for count, char in enumerate(word):
                # find the node path in tree or build node
                if node.child.get(char)!=None:
                    node = node.child.get(char)
                else:
                    new_node = Node(char)
                    node.child[char]=new_node
                    node = new_node
                # check whether it is a complete word or not
                if count == length-1:
                    node.count_back += freq
                    node.word_finish_back = True
                    
    def _search_1(self):
        node = self.root
        if not node.child:
            return
        total = 0
        for child in node.child.values():
            if child.word_finish == True:
                total += child.count
                total_l,total_r,p_l,p_r=0,0,0.0,0.0
                for ch in child.child.values():
                    if ch.word_finish_back:
                        total_l+=ch.count_back
                    if ch.word_finish:
                        total_r+=ch.count
                for ch in child.child.values():
                    if ch.word_finish_back:
                        p_l+=(ch.count_back / total_l) * math.log(ch.count_back / total_l, 2)
                    if ch.word_finish:
                        p_r+=(ch.count / total_r) * math.log(ch.count / total_r, 2)
                child.entropy = min(-p_l,-p_r)
                
        for child in node.child.values():
            if child.word_finish == True:
                child.probability = child.count / total

    def _search_2(self):
        node = self.root
        if not node.child:
            return 
        total = 0
        for child in node.child.values():
            for ch in child.child.values():
                if ch.word_finish == True:
                    total += ch.count
                    total_l,total_r,p_l,p_r=0,0,0.0,0.0
                    for gch in ch.child.values():
                        if gch.word_finish_back:
                            total_l+=gch.count_back
                        if gch.word_finish:
                            total_r+=gch.count
                    for gch in ch.child.values():
                        if gch.word_finish_back:
                            p_l+=(gch.count_back / total_l) * math.log(gch.count_back / total_l, 2)
                        if gch.word_finish:
                            p_r+=(gch.count / total_r) * math.log(gch.count / total_r, 2)
                    ch.entropy = min(-p_l,-p_r)
                    
        for child in node.child.values():
            for ch in child.child.values():
                if ch.word_finish == True :
                    ch.probability = ch.count / total
                    ch.PMI = math.log(ch.probability, 2) - math.log(child.probability, 2) - math.log(node.child[ch.char].probability, 2)

    def _search_3(self):
        node = self.root
        if not node.child:
            return 
        total = 0
        for child in node.child.values():
            for ch in child.child.values():
                for gch in ch.child.values():
                    if gch.word_finish == True:
                        total += gch.count
                        total_l,total_r,p_l,p_r=0,0,0.0,0.0
                        for ggch in gch.child.values():
                            if ggch.word_finish_back:
                                total_l+=ggch.count_back
                            if ggch.word_finish:
                                total_r+=ggch.count
                        for ggch in gch.child.values():
                            if ggch.word_finish_back:
                                p_l+=(ggch.count_back / total_l) * math.log(ggch.count_back / total_l, 2)
                            if ggch.word_finish:   
                                p_r+=(ggch.count / total_r) * math.log(ggch.count / total_r, 2)
                        gch.entropy=min(-p_l,-p_r)
                    
        for child in node.child.values():
            for ch in child.child.values():
                for gch in ch.child.values():
                    if gch.word_finish == True:
                        gch.probability = gch.count / total
                        temp_val = max(math.log(ch.probability, 2)+ math.log(node.child[gch.char].probability,2),math.log(child.probability, 2)+ math.log(node.child[ch.char].child[gch.char].probability,2))
                        gch.PMI = math.log(gch.probability, 2) - temp_val

    def _search_4(self):
        node = self.root
        if not node.child:
            return 
        total = 0
        for child in node.child.values():
            for ch in child.child.values():
                for gch in ch.child.values():
                    for ggch in gch.child.values():
                        if ggch.word_finish == True:
                            total += ggch.count
                            
                            total_l,total_r,p_l,p_r=0,0,0.0,0.0
                            for gggch in ggch.child.values():
                                if gggch.word_finish_back:
                                    total_l+=gggch.count_back
                                if gggch.word_finish:
                                    total_r+=gggch.count
                            for gggch in ggch.child.values():
                                if gggch.word_finish_back:
                                    p_l+=(gggch.count_back / total_l) * math.log(gggch.count_back / total_l, 2)
                                if gggch.word_finish:   
                                    p_r+=(gggch.count / total_r) * math.log(gggch.count / total_r, 2)
                            ggch.entropy=min(-p_l,-p_r)
                            
        for child in node.child.values():
            for ch in child.child.values():
                for gch in ch.child.values():
                    for ggch in gch.child.values():
                        if ggch.word_finish == True:
                            ggch.probability = ggch.count / total
                            temp_val = max(math.log(gch.probability, 2)+ math.log(node.child[ggch.char].probability,2), math.log(child.probability, 2)+ math.log(node.child[ch.char].child[gch.char].child[ggch.char].probability,2), math.log(ch.probability, 2)+math.log(node.child[gch.char].child[ggch.char].probability,2))
                            ggch.PMI = math.log(ggch.probability, 2) - temp_val
        return total
    
class TreeApp(Trietree):
    def __init__(self, char = '*', ngram_range = (1,4)):
        '''
        Parm corpus: List of different corpus.
        parm stopwords: Set of Vocabulary will be ignored.
        '''
        if not isinstance(ngram_range,tuple):
            if len(ngram_range) !=2:
                if ngram_range[0] > ngram_range[1] or ngram_range[1] not in [1,2,3,4] or ngram_range[0] not in [1,2,3,4]:
                    raise ValueError('Invalid Number of Gram Rage!!!')
                    
        super().__init__(char)
        self.ngram_range = ngram_range
        print('----------------Tree Initial Finished----------------', end = '\r')
        
    def _load_data_2_root(self,data):
        print('-----------------------------------------> Insert Node')
        for k, v in tqdm.tqdm(data.items()):
            self.add(k ,v)
        print('-----------------------------------------> Insert Finished')
    
    def add_new_data(self, corpus, stopwords = None):
        if not isinstance(corpus, list):
            raise ValueError('Must Input Corpus in List')
        if stopwords and not isinstance(stopwords, set):
            raise ValueError('Must Input Stopwords in Set')
        try:
            if stopwords:
                corpus = [[x for x in line.split() if x not in stopwords] for line in corpus]
            else :
                corpus = [line.split() for line in corpus]
            all_corp_ngram = [generate_ngram(x, n = self.ngram_range[1] + 1) for x in corpus]
            all_corp_ngram = [gram for line in all_corp_ngram for gram in line]
            all_corp_ngram.sort(key = len)
            grams = Counter(all_corp_ngram)
            self._load_data_2_root(grams)
        except Exception as e:
            raise ValueError('Failed to Add Data into Tree: '+ str(e))
            
    def set_treverse(self):
        for i in range(1,self.ngram_range[1]+1):
            try:
                fuc='self._search_'+str(i)+'()'
                eval(fuc)
                print('------------'+str(i)+'-gram Scoreing Treverse Finished------------', end = '\r')
            except Exception as e:
                raise ValueError('Failed to Set Node Score into Tree: '+ str(e))
            
    def get_treverse(self, PMI_bound = [5, 10, 15], entropy_bound = 1.0, frequency_bound = 10):
        if not isinstance(PMI_bound,list) and not isinstance(PMI_bound,int) and not isinstance(PMI_bound,float):
            raise ValueError("Not support PMI_bound as "+str(type(PMI_bound)))
        if not isinstance(entropy_bound,list) and not isinstance(entropy_bound,int) and not isinstance(entropy_bound,float):
            raise ValueError("Not support entropy_bound as "+str(type(entropy_bound)))
        if not isinstance(frequency_bound,list) and not isinstance(frequency_bound,int) and not isinstance(frequency_bound,float):
            raise ValueError("Not support frequency_bound as "+str(type(frequency_bound)))           
        if isinstance(PMI_bound,list):
            if (len(PMI_bound) != self.ngram_range[1] - self.ngram_range[0]) & (self.ngram_range[0] == 1):
                raise ValueError("PMI_bound lenth "+str(len(PMI_bound))+ "is not match with ngram_range "+str(self.ngram_range[1] - self.ngram_range[0]))
            elif (len(PMI_bound) == self.ngram_range[1] - self.ngram_range[0]) & (self.ngram_range[0] == 1):
                PMI_bound = [-1]+PMI_bound
            elif (len(PMI_bound) != self.ngram_range[1] - self.ngram_range[0] + 1) & (self.ngram_range[0] != 1):
                raise ValueError("PMI_bound lenth "+str(len(PMI_bound))+ "is not match with ngram_range "+str(self.ngram_range[1] - self.ngram_range[0] + 1))
            else:
                PMI_bound = [-1 for _ in range(self.ngram_range[0]-1)] + PMI_bound
        else:
            PMI_bound = [PMI_bound for _ in range(self.ngram_range[1])]
        if isinstance(entropy_bound,list):
            if (len(entropy_bound) != self.ngram_range[1] - self.ngram_range[0] + 1):
                raise ValueError("entropy_bound lenth "+str(len(entropy_bound))+ "is not match with ngram_range "+str(self.ngram_range[1] - self.ngram_range[0] + 1))
            else:
                entropy_bound = [-1 for _ in range(self.ngram_range[0]-1)] + entropy_bound
        else:
            entropy_bound = [entropy_bound for _ in range(self.ngram_range[1])]
        if isinstance(frequency_bound,list):
            if (len(frequency_bound) != self.ngram_range[1] - self.ngram_range[0] + 1):
                raise ValueError("frequency_bound lenth "+str(len(frequency_bound))+ "is not match with ngram_range "+str(self.ngram_range[1] - self.ngram_range[0] + 1))
            else:
                frequency_bound = [-1 for _ in range(self.ngram_range[0]-1)] + frequency_bound
        else:
            frequency_bound = [frequency_bound for _ in range(self.ngram_range[1])]   
        result = []
        node = self.root
        try:
            for child in node.child.values():
                if child.count > frequency_bound[0] and child.entropy > entropy_bound[0] and self.ngram_range[0] <= 1:
                    result.append((child.char, child.count, child.PMI, child.entropy, child.probability))
                if self.ngram_range[1] > 1:
                    for ch in child.child.values():
                        if ch.count > frequency_bound[1] and ch.entropy > entropy_bound[1] and ch.PMI > PMI_bound[1] and self.ngram_range[0] <= 2:
                            result.append((child.char+'_'+ch.char, ch.count, ch.PMI, ch.entropy, ch.probability))
                        if self.ngram_range[1] > 2:
                            for gch in ch.child.values():
                                if gch.count > frequency_bound[2] and gch.entropy > entropy_bound[2] and gch.PMI > PMI_bound[2] and self.ngram_range[0] <= 3:
                                    result.append((child.char+'_'+ch.char+'_'+gch.char, gch.count, gch.PMI, gch.entropy, gch.probability))
                                if self.ngram_range[1] > 3:
                                    for ggch in gch.child.values():
                                        if ggch.count > frequency_bound[3] and ggch.entropy > entropy_bound[3] and ggch.PMI > PMI_bound[3] and self.ngram_range[0] <= 4:
                                            result.append((child.char+'_'+ch.char+'_'+gch.char+'_'+ggch.char, ggch.count, ggch.PMI, ggch.entropy, ggch.probability))
        except Exception as e:
            raise ValueError('Failed to Retrieve Scores from Tree: '+ str(e))
        print('--------------Output to List of Tuple----------------', end = '\r')
        return result
        
    def phrase_mining(self, corpus, voc_candidates = None, stopwords = None, PMI_bound = [5, 10, 15], entropy_bound = 1.0, frequency_bound = 10):
        '''
        Parm corpus: List of different corpus.
        parm voc_candidates: Set of Vocabulary which all mining phrase have to contain one of it.
        parm stopwords: Set of Vocabulary will be ignored.
        parm PMI_bound: List or int or float of bound to PMI.
        parm entropy_bound: List or int or float of bound to entropy.
        parm frequency_bound: List or int or float of bound to frequency.
        Output: List of Tuple of frequency, PMI, entropy and probability of mining phrase.
        '''
        self.add_new_data(corpus, stopwords)
        self.set_treverse()
        result = self.get_treverse(PMI_bound, entropy_bound, frequency_bound)
        if voc_candidates:
            if not isinstance(voc_candidates, set):
                raise ValueError('Must Input voc_candidates in Set')
            result = [ (tup[0],tup[1],tup[2],tup[3],score_formulation(tup[2],tup[3],tup[4])) for tup in result if any([ j in voc_candidates for j in tup[0].split('_')])]
        else:
            result = [ (tup[0],tup[1],tup[2],tup[3],score_formulation(tup[2],tup[3],tup[4])) for tup in result]
        return result
        
    def pharse_scoring(self, corpus, phrase_list, stopwords = None):
        self.add_new_data(corpus, stopwords)
        self.set_treverse()
        if not isinstance(phrase_list, list):
            raise ValueError('Must Input phrase_list in List')
        result = [ (x,self.search(x)) for x in phrase_list]
        result = [ (tup[0],tup[1][0],tup[1][1],tup[1][2],score_formulation(tup[1][1],tup[1][2],tup[1][3])) for tup in result]
        return result