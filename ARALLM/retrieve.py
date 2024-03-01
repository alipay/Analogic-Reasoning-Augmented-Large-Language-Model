import numpy as np
import pandas as pd
import os

os.environ['CURL_CA_BUNDLE'] = ''
from sentence_transformers import SentenceTransformer,util

model = SentenceTransformer('BAAI/bge-large-zh',cache_folder="./retrieve_model",device='cuda')

reasoning_library = 'row_data/reasoning_library.csv'#指定推理库
rl_df = pd.read_csv(reasoning_library,encoding='utf-8')
rl_demand = rl_df['demand'].values.tolist()
rl_tag = rl_df['tag_list'].values.tolist()
rl_answer = rl_df['answer'].values.tolist()
rl_cot = rl_df['cot'].values.tolist()

demand_table = 'row_data/test_data.csv' #指定训练集或者测试集路径
text_df = pd.read_csv(demand_table,encoding='utf-8')
demand = text_df['demand'].values.tolist()
answer = text_df['answer'].values.tolist()

tag_table = 'row_data/tag_table.csv'#指定人群标签集，即key
tag_df = pd.read_csv(tag_table,encoding='utf-8')
tag = tag_df['tag_name'].drop_duplicates().values.tolist()

def retrieve_similar_demands_and_tags(demand,tag,rl_demand):
    demand_embeddings = model.encode(demand)
    tag_embeddings = model.encode(tag)
    rl_demand_embeddings = model.encode(rl_demand)

    d2t_sim = util.cos_sim(demand_embeddings,tag_embeddings)
    d2t_hits = util.semantic_search(demand_embeddings,tag_embeddings,top_k=15)

    d2d_sim = util.cos_sim(demand_embeddings,rl_demand_embeddings)
    d2d_hits = util.semantic_search(demand_embeddings,rl_demand_embeddings,top_k=3)

    results = []
    fixed_tag = [] #业务场景中的常用的标签，用于固定在标签列表中
    for i in range(len(d2t_hits)):
        result=[]
        q = demand[i]
        ans = answer[i]
        k_tag = []
        for hit in d2t_hits[i]:
            k_tag.append(tag[hit['corpus_id']].replace('，',' ').replace(',',' '))
        k_tag = fixed_tag + k_tag
        result.append(q)
        result.append(ans)
        result.append(','.join(k_tag))
        cnt = 0
        for rl_hits in d2d_hits[i]:
            if(rl_demand[hit['corpus_id']] != q):
                result.append(rl_demand[hit['corpus_id']])
                result.append(rl_tag[hit['corpus_id']])
                result.append(rl_cot[hit['corpus_id']])
                result.append(rl_answer[hit['corpus_id']])
                cnt += 1
            if(cnt==2): break
        results.append(result)
    final_data = pd.DataFrame(data=results,
                                        columns=['demand','answer','tag_list','example1','key_list1','cot1','ans1','example2','key_list2','cot2','ans2'])

    return final_data

def retrieve_similar_tags(demands,tags):
    demand_embeddings = model.encode(demand)
    tag_embeddings = model.encode(tag)

    sim = util.cos_sim(demand_embeddings,tag_embeddings)
    hits = util.semantic_search(demand_embeddings,tag_embeddings,top_k=15)

    results = []
    fixed_tag = [] #业务场景中的常用的标签，用于固定在标签列表中
    for i in range(len(hits)):
        result=[]
        q = demand[i]
        ans = answer[i]
        k_tag = []
        k_tag_score = []
        for hit in hits[i]:
            k_tag.append(tag[hit['corpus_id']].replace('，',' ').replace(',',' '))
            k_tag_score.append(hit['score'])
        k_tag = fixed_tag + k_tag
        result.append(q)
        result.append(ans)
        result.append(','.join(k_tag))
        result.append(','.join(str(s) for s in k_tag_score))
        results.append(result)
    demand_with_tag_list = pd.DataFrame(data=results,columns=['query','answer','tag_list','tag_score'])
    return demand_with_tag_list

final_data = retrieve_similar_demands_and_tags(demand,tag,rl_demand)