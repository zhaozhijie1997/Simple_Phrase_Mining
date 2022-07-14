import pandas as pd
import os
import sys
from multiprocessing.pool import ThreadPool
# from process_utils import process_sentence
import ahocorasick
from listing_commons.commons.phrases_mining.model import TreeApp
# from gensim.models import KeyedVectors
from collections import defaultdict
import argparse
from scipy.special import softmax
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
import numpy as np
import re
import requests
from polish import Polish

# module_path = os.path.abspath('/ldap_home/weiya.xu/xuannian_global_tree/keyword_mining/kw_mining_tagging')
# if module_path not in sys.path:
#     sys.path.append(module_path)

# from demos.tagging import AC_Matching
from attribute_tagging_offline import AC_Matching

import itertools


data_dir = "NER/data"
# cat = "fashion_accessories"
# market = "BR"
# target_att_type = 'Accessories Set'
w2v_path = "br_word_embedding_v1.vec"
# kw_count = 20

# L1 ID
# Attribute Type
# 1st Round KW (PIC:
# Attribute Value

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cat", help="category name", type=str)
    # parser.add_argument("-m", "--market", help="upper case", type=str)
    # parser.add_argument("-t", "--att_type", type=str)
    # parser.add_argument("-k", "--kw_count", type=int)
    return parser.parse_args()


def create_ac(dic):
    A = ahocorasick.Automaton()
    for i in dic:
        A.add_word(i,i)
    A.make_automaton()
    return A


def ac_words(ac,title):
    out = set()
    ts = title.split()
    for i,j in ac.iter(title):
        out.add(j)
    temp_1 = list(set(ts).intersection(out))
    temp_2 = [i for i in out if ' ' in i]
    temp_3 = [i for i in temp_2 if set(i.split()).issubset(ts) ]
    return temp_1+temp_3


if __name__ == "__main__":
    market = "PL"
    kw_count = 20


#   all the path
#   allegro_df_path: competitor data path
#   category_map_path: map local category to global category
#   att_df_path: all keywords for L1
#   shopee_df_path: shopee data path
    allegro_df_path = '/ldap_home/chiman.wong/git/resource/PL_attribute/PL_attribute/allegro_data'
    category_map_path = 'category_mapping.csv'
    att_df_path = 'PL_keyword_0422.csv'
    shopee_df_path = '/ldap_home/chiman.wong/git/resource/PL_attribute/PL_attribute/local_data/PL_local_data.csv'
    
#   get global category mapping
    category_map_df = pd.read_csv(category_map_path)
    cate_map = {}
    for index, data in category_map_df.iterrows():
        cate_map[data['MY']] = data['Leaf Node ID']
    
#   get shopee PL dta
    shopee_df = pd.read_csv(shopee_df_path)
    
#   preprocess PL title
    pl = Polish()
    shopee_df['preprocessed_title'] = shopee_df['title'].apply(lambda x:pl.preprocess(x))
    shopee_df = shopee_df[['itemid', 'title', 'preprocessed_title', 'global_L1_catid', 'end_leaf_catid']]

#   iterate all competitor files
    for filename in  os.listdir(allegro_df_path):
        print(os.path.join(allegro_df_path, filename))
        cat = filename.split('_')[0]

        if not os.path.exists('result/'+str(cat)):
            os.makedirs('result/'+str(cat), exist_ok=True)

#       merge shopee data with competitor data  
        allegro_df = pd.read_csv(os.path.join(allegro_df_path, filename))
        allegro_df = allegro_df[['itemid', 'title', 'preprocessed_title', 'global_L1_catid', 'end_leaf_catid']]
        all_item_df = pd.concat([allegro_df, shopee_df], ignore_index=True)

        result=[]
        
        for attr_type in  list(set(attribute_df[attribute_df['L1 ID'] == cat]['Attr_Name_DS'].values)):
            print("process attr_type: " + str(attr_type))
            target_att_type = attr_type
            
            kw_count_out_path = os.path.join('result/'+str(cat), attr_type,  'kw_mining_count.csv')
            kw_cluster_output_path= os.path.join('result/'+str(cat), attr_type,  'clusters.csv')
            kw_output_path = os.path.join('result/'+str(cat), attr_type, 'kw_clusters.csv')

#           check if the files already processed  
            if type(attr_type) == float and os.path.exists(os.path.join('result/'+str(cat), attr_type,  'clusters.csv'):
                continue

            # if attr_type.find('Length') != -1 or attr_type.find('Size') != -1:
            #     continue

        
            if '/' in attr_type or ',' in attr_type:
                attr_type = attr_type.replace('/',' ').replace(',',' ')

            print("after preprocessing, attr_type is  " + str(attr_type))


            if not os.path.exists('result/' + str(cat) + '/' + attr_type):
                os.makedirs('result/' + str(cat) + '/' + attr_type, exist_ok=True)

#           get relevent cateID for that type
            attribute_df = pd.read_csv(att_df_path)
            attribute_df = attribute_df[~attribute_df[f"Collection ID_value"].isna()]
            temp_id = list(set(attribute_df[attribute_df['Attr_Name_DS']==target_att_type]['Collection ID_value'].values))
            collection_id = []

            if temp_id == []:
                continue
            for i in temp_id:
                for j in i.split(','):
                    if j.strip() is not None and len(j)>2 and float(j.strip()) in cate_map.keys():
                        collection_id.extend([cate_map[float(j.strip())]])
            collection_id = [int(i) for i in collection_id]
            item_df = all_item_df[all_item_df['end_leaf_catid'].isin(collection_id)]
            item_df = item_df.drop_duplicates(subset=['itemid'])
            print(f"All global cat ids: {collection_id}")
            print(f"Item number under corresponding global_cat ids: {len(item_df)}")
                                                           
            info_df = item_df.copy()
            info_df = info_df.drop_duplicates()
            pool = ThreadPool(20)

            target_title = 'preprocessed_title'
            PMI_threshold = 2.5
            entropy = [0.5,0.5,0.5]
            freq_threshold = [20, 15, 10]
            #
            sin_kw2score, sin_kw2att, sin_kw2fre = defaultdict(list), defaultdict(list), defaultdict(list)
            mul_kw2score, mul_kw2att, mul_kw2fre = defaultdict(list), defaultdict(list), defaultdict(list)

            temp_df = info_df.copy()
            all_title = list(temp_df[target_title].astype(str).values)
            tree_application = TreeApp(ngram_range = (1,3))
            result = tree_application.phrase_mining(all_title, None, None, PMI_threshold, entropy, freq_threshold)
            output = pd.DataFrame(result, columns=['keywords', 'frequency', 'PMI', 'entropy', 'score'])

            single_temp_df = output[output.score.isna()]
            multi_temp_df = output[~output.score.isna()]
            singl_sum_freq = sum(single_temp_df.frequency.values)
            single_temp_df['score'] = single_temp_df.entropy*(single_temp_df.frequency/singl_sum_freq)
            for x, y, z in zip(single_temp_df.keywords,single_temp_df.score,single_temp_df.frequency):
                sin_kw2score[x].append(y)
                sin_kw2fre[x].append(z)
                sin_kw2att[x].append(target_att_type)
            for x, y, z in zip(multi_temp_df.keywords,multi_temp_df.score, multi_temp_df.frequency):
                mul_kw2score[x].append(y)
                mul_kw2fre[x].append(z)
                mul_kw2att[x].append(target_att_type)
            #     print('Attribute Type {} Finished, {}% of Process Finished' .format(str(i), str(((ind+1)/len(info_df['att_type'].unique()))*100)))
            # Single Words
            sin_kw2score_2, sin_kw2att_2, sin_kw2fre_2 = dict(), dict(), dict()
            for k,v in  sin_kw2score.items():
                prob_v = softmax(v)
                sin_kw2score_2[k] = np.max(prob_v)*v[np.argmax(prob_v)]
                sin_kw2fre_2[k] = sin_kw2fre[k][np.argmax(prob_v)]
                sin_kw2att_2[k] = sin_kw2att[k][np.argmax(prob_v)]
            sin_kw2fre_2 = {k: v for k, v in sorted(sin_kw2fre_2.items(), key=lambda item: item[1],reverse = True)}
            single_df = pd.DataFrame()
            single_df['keywords'] = sin_kw2fre_2.keys()
            single_df['frequency'] = sin_kw2fre_2.values()
            single_df['score'] = sin_kw2score_2.values()
            single_df['att_type'] = single_df['keywords'].apply(lambda x: sin_kw2att_2[x])
            # Multi Words
            mul_kw2score_2, mul_kw2att_2, mul_kw2fre_2 = dict(), dict(), dict()
            for k,v in  mul_kw2score.items():
                prob_v = softmax(v)
                mul_kw2score_2[k] = np.max(prob_v)*v[np.argmax(prob_v)]
                mul_kw2fre_2[k] = mul_kw2fre[k][np.argmax(prob_v)]
                mul_kw2att_2[k] = mul_kw2att[k][np.argmax(prob_v)]
            mul_kw2fre_2 = {k: v for k, v in sorted(mul_kw2fre_2.items(), key=lambda item: item[1],reverse = True)}
            multi_df = pd.DataFrame()
            multi_df['keywords'] = mul_kw2score_2.keys()
            multi_df['frequency'] = mul_kw2fre_2.values()
            multi_df['score'] = mul_kw2score_2.values()
            multi_df['att_type'] = multi_df['keywords'].apply(lambda x: mul_kw2att_2[x])
            multi_df.keywords = multi_df.keywords.apply(lambda x: str(x).replace('_',' ').strip())

            pool = ThreadPool(20)

            # Tagging For Filter
            single_ac_solver = AC_Matching(single_df.keywords.values)
            info_df = info_df.assign(single_tagging = pool.map(lambda x: set(single_ac_solver.exact_match(str(x))), info_df[target_title]))
            print('Single Tagging Finished')
            multi_ac_solver = AC_Matching(multi_df.keywords.values)
            info_df = info_df.assign(multi_tagging = pool.map(lambda x: set(multi_ac_solver.exact_match(str(x))), info_df[target_title]))
            print('Multi Tagging Finished')

            # Choose Threshold
            batch = len(single_df)
            length = 20
            info_df_len = len(info_df)
            single_record, multi_record = [], []

            for i in range(length):
                sigle_set = set(single_df.head(batch*(i+1)).keywords.values)
                CE = len(info_df[info_df.single_tagging.apply(lambda x: len(x.intersection(sigle_set))) > 0])/info_df_len
                if (len(single_record) > 0):
                    if (CE - single_record[-1] <= 0.01):
                        print('Single Ketwords Size {}, CE Rate {}%'.format(str(len(single_record)*batch),str(single_record[-1]*100)))
                        break
                single_record.append(CE)
                print((i+1)/length,end = '\r')
            single_df = single_df.head(len(single_record)*batch)

            for i in range(length):
                multi_set = set(multi_df.head(batch*(i+1)).keywords.values)
                CE = len(info_df[info_df.multi_tagging.apply(lambda x: len(x.intersection(multi_set))) > 0])/info_df_len
                if (len(multi_record) > 0):
                    if(CE - multi_record[-1] <= 0.01):
                        print('Multi Ketwords Size {}, CE Rate {}%'.format(str(len(multi_record)*batch),str(multi_record[-1]*100)))
                        break
                multi_record.append(CE)
                print((i+1)/length,end = '\r')
            multi_df = multi_df.head(len(multi_record)*batch)
            whole_df = multi_df.append(single_df)
            whole_df['keywords_len'] = whole_df.keywords.apply(lambda x: len(x.split()))
            # phrase_path = os.path.join('20200820_health_att_kw_mining.csv')
            # whole_df.to_csv(phrase_path, index = False)

            print(f">>>>>>>>> all keywords (phrases) number: {len(whole_df)}")
            pool = ThreadPool(20)

            phrases = list(whole_df['keywords'].unique())
            ac = create_ac(phrases)

            item_df = item_df.assign(title_new_keywords_found = pool.map(lambda x: ac_words(ac,str(x)), item_df.preprocessed_title))

            #get keyword count

            from collections import Counter
            from itertools import chain

            kw_list_sub = list(item_df['title_new_keywords_found'].values)
            kw_list_sub = list(chain(*kw_list_sub))
            kw_freq = Counter(kw_list_sub).most_common()
            tmp_kw_sub_df = pd.DataFrame(kw_freq, columns=['keywords', 'count'])
            tmp_kw_sub_df = tmp_kw_sub_df[tmp_kw_sub_df['keywords']!='']

            result = pd.merge(whole_df.drop(columns=['frequency']),
                                tmp_kw_sub_df,
                                on=['keywords'],
                                how='outer')
            result = result.dropna(subset=['count'])

            # get initial keywords
            if market not in attribute_df.columns:
                attribute_df = attribute_df.rename(columns={f"Attribute Value(pl)": market})
            attribute_df = attribute_df.dropna(subset=[market])
            attribute_df = attribute_df[['Attr_Name_DS', 'Value_Name_DS', market]]

            attribute_df[market] = attribute_df[market].apply(lambda x: re.split(r"[,，] *|\n", x.strip()))
            # for index, content in attribute_df.iterrows():
            #     kws = content[market]
            #     attribute_df.loc[index, market] = re.split(r"[,，] *|\n", kws.strip())

            kws = []
            kws_dict = {}
            for _, content in attribute_df.iterrows():

                kw = [i.strip() for i in content[market]]

                kws.extend(kw)
                for w in kw:
                    if w != '':
                        kws_dict[w] = [content['Attr_Name_DS'], content['Value_Name_DS']]
            kws[:] = [x for x in kws if x]
            kws = list(set(kws))
            pl = Polish()
            kws_tok = [pl.preprocess(i) for i in kws]

            ac = create_ac(kws_tok)
            result = result.assign(contains_initial_keywords = pool.map(lambda x: ac_words(ac,str(x)), result.keywords))
            result['contains_initial_keywords'] = result['contains_initial_keywords'].apply(lambda x: '&&'.join(x))

            result.to_csv(kw_count_out_path, index=False)

            # Keyword Clustering
            # word_vectors = KeyedVectors.load_word2vec_format(w2v_path, binary=False)  # C text format
            keywords_df = result.copy()
            # to make keyword number in [1500, 5000]
            while kw_count > 10:
                tmp = keywords_df[keywords_df['count']>kw_count]
                if 1500 <= len(tmp) <= 5000:
                    break
                if len(tmp) < 1500:
                    kw_count -= 1
                if len(tmp) > 5000:
                    kw_count += max(1, int((len(tmp) - 5000) / 100))

            keywords_df = keywords_df[keywords_df['count']>kw_count] #choose keywords with count > 50
            keywords = list(keywords_df['keywords'].unique())
            print(f"number of keywords with kw count > {kw_count}: {len(keywords)}")

            #get word embedding for keywords
            vector = {}

            try:
#               bert labcse                                                      
                data_vec=requests.post("http://10.131.23.6:9111/text_encode", json={'country': 'BASE', 'texts': keywords, 'batch_size': 128}).json()
            except:
                print('request failed')

            # for i in keywords:
            for count, keyword in enumerate(keywords):

                # keyword = i
                vector[keyword] = data_vec[count]
                # try:
                #     if " " in keyword:
                #         words = keyword.split(' ')
                #         num_words = len(words)
                #
                #         vector[keyword] = requests.post("http://10.131.23.6:9111/text_encojson={'country': 'BASE', 'texts': ["hello hi I am peter"],'batch_size': 100}).json()
                #
                #         # for word in words:
                #         #     # if only digit, then dump
                #         #     if keyword in vector:
                #         #         vector[keyword] = word_vectors[word] + vector[keyword] #use original vectors
                #         #         #vector[keyword] = word_vectors.word_vec(word,use_norm=True) + vector[keyword] #use normalized vectors
                #         #     else:
                #         #         vector[keyword] = word_vectors[word]
                #         #         #vector[keyword] = word_vectors.word_vec(word,use_norm=True)
                #         vector_sum =  vector[keyword]
                #         vector[keyword] = vector_sum / num_words
                #     else:
                #         vector[keyword] = word_vectors[keyword]
                # except:
                #     print('out of vocabulary:', i)
                #     continue

            print('vec complete ...')
            vector_dict = vector
            vector = pd.DataFrame.from_dict(vector, orient='index')

            #prepare training data for K-means
            X = []
            words = []
            for key,value in vector_dict.items():
                # if len(value) != 300:
                #     print(value)
                #     print('*'*30)
                #     continue
                X.append(value)
                words.append(key)

            min_k = 20
            max_k = 80
            init_algo = 'k-means++'

            #Compute the Silhouette Value
            sil = []
            ss = []
            max_sil = -1
            max_sil_idx = 0

            # if
            for k in range(min_k, max_k+1):
                if k > len(X)-1:
                    break
                kmeans = KMeans(n_clusters = k, init = init_algo, n_init = 5, random_state=42)
                kmeans.fit(X)
                labels = kmeans.labels_

                sil.append(silhouette_score(X,labels,metric='euclidean'))
                ss.append(kmeans.inertia_)
                if sil[-1] > max_sil:
                    max_sil = sil[-1]
                    max_sil_idx = k
                print(f"Clustering k={k}, score={sil[-1]}", end='\r', flush=True)


            # #plot within-cluster sum-of-squares
            # f, axes = plt.subplots(1,1,figsize=(16,4))
            # plt.plot(range(min_k,max_k+1),ss)
            # plt.xlabel('Number of Clusters')
            # plt.ylabel('Sum of Squares Within clusters')
            # plt.xticks(np.arange(min_k,max_k+1,1.0))
            # plt.grid(which='major',axis='y')
            # plt.show()

            # #plot Silhouette score
            # f, axes = plt.subplots(1,1,figsize=(16,4))
            # plt.plot(range(min_k,max_k+1),sil)
            # plt.xlabel('Number of Clusters')
            # plt.ylabel('Silhouette Score')
            # plt.xticks(np.arange(min_k,max_k+1,1.0))
            # plt.grid(which='major',axis='y')
            # plt.show()

            NUM_CLUSTERS = max_sil_idx
            print(f"Best cluster number: {NUM_CLUSTERS} at sli score of {max_sil}")
            # dbscan = DBSCAN(1)
            # labels = dbscan.fit_predict(X)


            kmeans = KMeans(n_clusters = NUM_CLUSTERS, init = init_algo, n_init = 5, random_state=42)
            kmeans.fit(X)
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_
            print("Cluster id labels for inputted data")
            print(labels)
            print("Centroids data")
            print(centroids)

            print(
                "Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
            print(kmeans.score(X))

            silhouette_score_result = silhouette_score(X, labels, metric='euclidean')

            print("Silhouette_score: ")
            print(silhouette_score_result)

            model = TSNE(n_components=2, random_state=0)
            np.set_printoptions(suppress=True)

            Y = model.fit_transform(X)
            # plt.scatter(Y[:, 0], Y[:, 1], c=labels, s=290, alpha=.5)
            # plt.show()

            word_label_dict = {'cluster_label':[],'keywords':[]}

            for i in range(len(labels)):
                if labels[i] in word_label_dict['cluster_label']:
                    ind = word_label_dict['cluster_label'].index(labels[i])
                    word_label_dict['keywords'][ind] = word_label_dict['keywords'][ind]+', '+words[i]
                else:
                    word_label_dict['cluster_label'].append(labels[i])
                    word_label_dict['keywords'].append(words[i])

            clusters_df = pd.DataFrame.from_dict(word_label_dict)
            clusters_df.to_csv(kw_cluster_output_path, index=False)

            # assign keywords to clusters

            clusters_dict = {}

            for _, content in clusters_df.iterrows():
                kws = content['keywords']
                kws = kws.split(', ')
                for kw in kws:
                    clusters_dict[kw] = content['cluster_label']

            cluster_labels = []

            for _, content in keywords_df.iterrows():
                if content['count'] < kw_count:
                    cluster_labels.append('low frequency keyword')
                elif content['keywords'] not in clusters_dict:
                    cluster_labels.append('oov')
                else:
                    cluster_labels.append(clusters_dict[content['keywords']])

            keywords_df['cluster_label'] = cluster_labels

            clustered_keywords_df = keywords_df[~keywords_df['cluster_label'].isin(['low frequency keyword','oov'])]

            #sort keywords by count within each cluster
            clustered_keywords_df = clustered_keywords_df.sort_values(by=['cluster_label','count'],ascending=False)
            clustered_keywords_df.to_csv(kw_output_path, index=False)

            print('done ...')