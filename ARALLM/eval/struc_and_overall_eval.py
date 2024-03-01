import evaluate
import difflib
import Levenshtein
import pandas as pd


def filter_string(string):
    filtered_string = ''
    i = 0
    while i < len(string):
        if string[i] == '(' or string[i] == ')' or string[i] == ' ':
            filtered_string += string[i]
            i += 1
        elif string[i:i+3] == 'AND':
            filtered_string += 'AND'
            i += 3
        elif string[i:i+2] == 'OR':
            filtered_string += 'OR'
            i += 2
        elif string[i] == '#':
            filtered_string += '#'
            i += 1
        else:
            i += 1
    return filtered_string

# import your predict results and label
result_table = ''
result_df = pd.read_csv(result_table,encoding='utf-8')

label = result_df['label']
struc_label = [filter_string(x) for x in result_df['label']]

predict = result_df['predict']
struc_pre = [filter_string(x) for x in predict]


def levenshtein_similarity(pred, gold):
    if not gold:
        # If gold is empty, give zero score
        return 0.0
    return 1 - Levenshtein.distance(pred, gold) / (2 * len(gold))

def difflib_similarity(pred, gold):
    sm = difflib.SequenceMatcher(None, pred, gold)
    return sm.ratio()

#rouge and sacrebleu
rouge = evaluate.load('./evaluate_local/metrics/rouge/rouge.py')
sacrebleu = evaluate.load("./evaluate_local/metrics/sacrebleu/sacrebleu.py")


# rouge_results = rouge.compute(predictions=predict, references=label)['rougeL']
# print('rouge_results:', rouge_results)

sacrebleu_results = sacrebleu.compute(predictions=predict, references=label)["score"]
print('sacrebleu_results', round(sacrebleu_results,1))

levenshtein_score = []
diff_lib_sim = []
for j in range(len(struc_pre)):
    levenshtein_score.append(levenshtein_similarity(struc_pre[j],struc_label[j]))
    diff_lib_sim.append(difflib_similarity(struc_pre[j],struc_label[j]))
print('L:', sum(levenshtein_score)/len(levenshtein_score))
print('R/O:', sum(diff_lib_sim)/len(diff_lib_sim))
print('Mean:',(sum(levenshtein_score)+sum(diff_lib_sim))/(2*len(levenshtein_score)))
        
#bertscore
# bertscore = evaluate.load("./evaluate/metrics/bertscore/bertscore.py")
# bertscore_results = bertscore.compute(predictions=pre, references=ref, lang="en")['f1']
# print('bertscore:', bertscore)

#bleurt
# bleurt = evaluate.load("./evaluate/metrics/bleurt/bleurt.py", module_type="metric")
# bleurt_results = bleurt.compute(predictions=pre, references=ref)['scores']
# print('bleurt_results', bleurt_results)

