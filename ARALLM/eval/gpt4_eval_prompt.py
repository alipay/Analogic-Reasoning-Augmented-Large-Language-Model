import pandas as pd

instruction = str()
with open('gpt4_eval_instruction.txt','r') as f:
    instruction = f.read()

# import your predict results and label
result_table = ''
result_df = pd.read_csv(result_table,encoding='utf-8')

predict = result_df['predict']
label = result_df['label']

eval_prompt_list = []
for i, ans in enumerate(predict):
    prompt = instruction

    prompt = prompt + 'Predicted answer: ' + ans + '\n'
    prompt = prompt + 'Ground truth answer: ' + label[i] + '\n'
    prompt = prompt + 'Accuracy: '
    eval_prompt_list.append(prompt)