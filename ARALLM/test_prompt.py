import torch
import numpy as np
import pandas as pd
import random
from .retrieve import retrieve_similar_demands_and_tags
#basic instruction
instruction = str()
with open('instruction.txt','r') as f:
    instruction = f.readline()

#load reasoning_library
reasoning_library = 'row_data/reasoning_library.csv'
rl_df = pd.read_csv(reasoning_library,encoding='utf-8')
rl_demand = rl_df['demand'].values.tolist()
rl_tag = rl_df['tag_list'].values.tolist()
rl_answer = rl_df['answer'].values.tolist()
rl_cot = rl_df['cot'].values.tolist()


def zero_shot_prompt(demand_data):
    demand_list = demand_data['demand'].values.tolist()
    tag_list = demand_data['tag_list'].values.tolist()
    prompt_list = []
    for idx in range(len(demand_list)):
        prompt = instruction + '\n' + 'query:\n' + demand_list[idx] + \
                 '\n' + 'key list:' + tag_list[idx] + 'answer:\n'
    prompt_list.append(prompt_list)

    return prompt_list

def fixed_few_shot_without_cot_prompt(demand_data):
    instruction = instruction + '请阅读以下例子并回答。'
    demand_list = demand_data['demand'].values.tolist()
    tag_list = demand_data['tag_list'].values.tolist()
    prompt_list = []
    for idx in range(len(demand_list)):
        prompt = instruction + '\n'
        prompt = prompt + '###\n'
        prompt = prompt + 'demand:\n' + rl_demand[0] + '\n' + 'key list:\n' + '[' + rl_tag[0] + ']\n' + 'answer:\n' + rl_answer[0] + '\n'
        prompt = prompt + '###\n'
        prompt = prompt + 'demand:\n' + rl_demand[1] + '\n' + 'key list:\n' + '[' + rl_tag[1] + ']\n' + 'answer:\n' + rl_answer[1] + '\n'
        prompt = prompt + '###\n'
        prompt = prompt + 'demand:\n' + demand_list[idx] + '\n' + 'key list:\n' + '[' + tag_list[idx] + ']\n' + 'answer:\n'
        prompt_list.append(prompt)
    
    return prompt_list

def fixed_few_shot_with_cot_prompt(demand_data):
    instruction = instruction + '请阅读以下例子并回答。'
    demand_list = demand_data['demand'].values.tolist()
    tag_list = demand_data['tag_list'].values.tolist()
    prompt_list = []
    for idx in range(len(demand_list)):
        prompt = instruction + '\n'
        prompt = prompt + '###\n'
        prompt = prompt + 'demand:\n' + rl_demand[0] + '\n' + \
                 'key list:\n' + '[' + rl_tag[0] + ']\n' + \
                 'reasoning:\n' + rl_cot[0] + '\n' + \
                 'answer:\n' + rl_answer[0] + '\n'
        prompt = prompt + '###\n'
        prompt = prompt + 'demand:\n' + rl_demand[1] + '\n' + \
                 'key list:\n' + '[' + rl_tag[1] + ']\n' + \
                 'reasoning:\n' + rl_cot[1] + '\n' + \
                 'answer:\n' + rl_answer[1] + '\n'
        prompt = prompt + '###\n'
        prompt = prompt + 'demand:\n' + demand_list[idx] + '\n' + 'key list:\n' + '[' + tag_list[idx] + ']\n' + 'answer:\n'
        prompt_list.append(prompt)
    
    return prompt_list

def random_few_shot_prompt(demand_data):
    instruction = instruction + '请阅读以下例子并回答。'

    demand_list = demand_data['demand'].values.tolist()
    tag_list = demand_data['tag_list'].values.tolist()
    prompt_list = []

    for idx in range(len(demand_list)):
        seed_idx = random.sample(range(0,len(rl_demand)),2)
    
        ex1_d = rl_demand[seed_idx[0]]
        tag1 = rl_tag[seed_idx[0]]
        cot1 = rl_cot[seed_idx[0]]
        ans1 = rl_answer[seed_idx[0]]
        
        ex2_d = rl_demand[seed_idx[1]]
        tag2 = rl_tag[seed_idx[1]]
        cot2 = rl_cot[seed_idx[1]]
        ans2 = rl_answer[seed_idx[1]]

        prompt = instruction + '\n'
        prompt = prompt + '###\n'
        prompt = prompt + 'query:\n' + ex1_d[idx] + '\n' + \
                'key list:\n' + '[' + tag1[idx] + ']\n' + \
                'reasoning:\n' + cot1[idx] + '\n' + \
                'answer:\n' + ans1[idx] + '\n'
        prompt = prompt + '###\n'
        prompt = prompt + 'query:\n' + ex2_d[idx] + '\n' + \
                'key list:\n' + '[' + tag2[idx] + ']\n' + \
                'reasoning:\n' + cot2[idx] + '\n' + \
                'answer:\n' + ans2[idx] + '\n'
        prompt = prompt + '###\n'
        prompt = prompt + 'query:\n' + demand_list[idx] + '\n' + 'key list:\n' + '[' + tag_list[idx] + ']\n' + 'answer:\n'
        prompt_list.append(prompt)

    return prompt_list

def ara_prompt(demand_data):
    instruction = instruction + '请阅读以下例子并回答。'

    demand_list = demand_data['demand'].values.tolist()
    tag_list = demand_data['tag_list'].values.tolist()
    prompt_list = []
    ex1_d = demand_data['example1']
    tag1 = demand_data['key_list1']
    cot1 = demand_data['cot1']
    ans1 = demand_data['ans1']
    ex2_d = demand_data['example2']
    tag2 = demand_data['key_list2']
    cot2 = demand_data['cot2']
    ans2 = demand_data['ans2']

    prompt_list = []
    for idx in range(len(demand_list)):
        prompt = instruction + '\n'
        prompt = prompt + '###\n'
        prompt = prompt + 'query:\n' + ex1_d[idx] + '\n' + \
                'key list:\n' + '[' + tag1[idx] + ']\n' + \
                'reasoning:\n' + cot1[idx] + '\n' + \
                'answer:\n' + ans1[idx] + '\n'
        prompt = prompt + '###\n'
        prompt = prompt + 'query:\n' + ex2_d[idx] + '\n' + \
                'key list:\n' + '[' + tag2[idx] + ']\n' + \
                'reasoning:\n' + cot2[idx] + '\n' + \
                'answer:\n' + ans2[idx] + '\n'
        prompt = prompt + '###\n'
        prompt = prompt + 'query:\n' + demand_list[idx] + '\n' + 'key list:\n' + '[' + tag_list[idx] + ']\n' + 'answer:\n'
        prompt_list.append(prompt)
    
    return prompt_list


demand_table = 'row_data/test_data.csv' # test dataset
text_df = pd.read_csv(demand_table,encoding='utf-8')
demand = text_df['demand'].values.tolist()
answer = text_df['answer'].values.tolist()

tag_table = 'row_data/tag_table.csv' # tag list
tag_df = pd.read_csv(tag_table,encoding='utf-8')
tag = tag_df['tag_name'].drop_duplicates().values.tolist()

demand_data = retrieve_similar_demands_and_tags(demand=demand,tag=tag,rl_demand=rl_demand)
ara_p = ara_prompt(demand_data) 
# You can also generate other versions of prompt just change the function name.