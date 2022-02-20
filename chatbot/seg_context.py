import preprocess as ps
import context_list as clist
from sentence_transformers import SentenceTransformer
import json
import re

delimiters = ['?', '!', ';', '？', '！', '。', '；', '……', '…', '\n', ';']
context_list = clist.context_list
chinese1 = '[^\u4e00-\u9fa5]+'

context_list_split_by_delimiters = {
    'text_data': []
}
s_bert_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

for idx in range(len(context_list)):
    temp_list = ps.segment(context_list[idx], delimiters)
    for jdx in range(len(temp_list)):
        # seg_sentence = ''.join(ps.seg_sentence(temp_list[jdx], ps.stopwords)),
        seg_sentence =temp_list[jdx]
        # print(seg_sentence)
        context_list_split_by_delimiters['text_data'].append({
            'context': temp_list[jdx],
            'context_idx': idx,
            'seg': seg_sentence,
            'encode': s_bert_model.encode(re.sub(chinese1, "", seg_sentence)).tolist()
        })

with open('seg_context.json', 'w+') as outfile:
    json.dump(context_list_split_by_delimiters, outfile)