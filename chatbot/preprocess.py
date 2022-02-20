# -- coding:UTF-8 --
import torch
import jieba
import re
import json

# 載入繁體辭典
jieba.set_dictionary('./dict.txt.big')

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords  

# 對句子進行分詞  
def seg_sentence(sentence, stopwords):  
    sentence = re.sub('[^\u4e00-\u9fa5a-zA-Z]+','',sentence)
    sentence_seged = jieba.cut(sentence.strip())  
      
    outstr = []
    for word in sentence_seged:  
        if word not in stopwords:  
            if word != '\t' and word != ' ':  
                outstr.append(word)
    return outstr  

def add_token_positions(encodings, answers, tokenizer):
    # initialize lists to contain the token indices of answer start/end
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length

        shift = 1
        while end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - shift)
            shift += 1

    if len(start_positions) == 0:
      start_positions.append(512)
    if len(end_positions) == 0:
      end_positions.append(512)
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        # print(self.encodings)
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

# 斷句測試，對context list裡進行斷詞
def segment(text, delimiters):
    res = [text] 
    for sep in delimiters:
        text, res = res, []
        for seq in text:
            res += seq.split(sep)
    res = [s.strip() for s in res if len(s.strip()) > 0]
    return res

def load_seg_context_embedding(data_path: str):
    with open(data_path) as f:
        json_data = json.load(f)
    
    embedding_data = json_data['text_data']

    return embedding_data


stopwords = stopwordslist('stopword.txt')
seg_context_embedding = load_seg_context_embedding('seg_context.json')