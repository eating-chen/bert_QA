# -- coding:UTF-8 --
from transformers import BertConfig
from torch.utils.data import DataLoader
from transformers import BertForQuestionAnswering
import torch
from transformers import BertTokenizerFast
import preprocess as ps
import context_list as clist
from sentence_transformers import SentenceTransformer, util
import numpy as np
from gensim.summarization import bm25
import heapq
import re
from fuzzychinese import FuzzyChineseMatch

chinese1 = '[^\u4e00-\u9fa5]+'

def answer_question(qn, context, bert_tokenizer, bertQAModel, device):
    bertQAModel.eval()
    encodings = bert_tokenizer(context, qn, truncation=True, padding=True)
    ps.add_token_positions(encodings, '', bert_tokenizer)

    val_dataset = ps.SquadDataset(encodings)
    val_loader = DataLoader(val_dataset, batch_size=1)
    val_data = next(iter(val_loader))
    answer = ''
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        input_ids = val_data['input_ids'].to(device)
        attention_mask = val_data['attention_mask'].to(device)

        outputs = bertQAModel(input_ids, attention_mask=attention_mask)

        start_logits_softmax = softmax(outputs['start_logits'])
        start_pred = torch.argmax(start_logits_softmax, dim=1)
        start_score = torch.max(start_logits_softmax, dim=1)[0].item()

        end_logits_softmax = softmax(outputs['end_logits'])
        end_pred = torch.argmax(end_logits_softmax, dim=1)
        end_score = torch.max(end_logits_softmax, dim=1)[0].item()

        average_socre = (start_score+end_score)/2
        tokens = bert_tokenizer.convert_ids_to_tokens(input_ids[0])
        answer=tokens[start_pred]
        if end_pred <= start_pred:
            answer = '輸入其他問題試試?'
            average_socre = 0
        else:
            for i in range(start_pred+1, end_pred):
                tokens[i] = tokens[i].replace('##', '')
                answer += tokens[i]
        
    return {
        'answer': answer,
        'average_socre': average_socre,
        'start_score': start_score,
        'end_score': end_score
    }

def load_model(epoch: str):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    jimmy_QA_config = BertConfig.from_json_file('./model/jimmy_QA_config_file'+epoch+'.bin')
    jimmy_QA_model = BertForQuestionAnswering(jimmy_QA_config).to(device)
    jimmy_QA_model.load_state_dict(torch.load('./model/jimmy_QA_model'+epoch+'.bin'))
    jimmy_QA_tokenizer = BertTokenizerFast('./model/vocab.txt')
    s_bert_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
    # s_bert_model = None
    print('load finish!!')

    return {
        'model': jimmy_QA_model,
        's_bert_model':s_bert_model,
        'tokenizer': jimmy_QA_tokenizer,
        'device': device
    }

def compare_all_context(question, tokenizer, model, device, sbert_model):
    context_list_split_by_delimiters = ps.seg_context_embedding
    # question_str = ''.join(ps.seg_sentence(question, ps.stopwords))
    question_str = re.sub(chinese1, "", question)
    question_str = question
    print('question: ', question_str)
    emb1 = sbert_model.encode(question_str)

    s_bert_res_list = []
    for idx in range(len(context_list_split_by_delimiters)):
        # emb2 = sbert_model.encode(context_list_split_by_delimiters[idx]['context'])
        # print(context_list_split_by_delimiters[idx]['encode'])
        cos_sim = util.cos_sim(emb1, context_list_split_by_delimiters[idx]['encode'])
        s_bert_res_list.append({
            "seg_context": context_list_split_by_delimiters[idx]['context'], 
            "CosineSimilarity": cos_sim.item(),
            "context_idx": context_list_split_by_delimiters[idx]['context_idx']
        })
    
    s_bert_res_list.sort(key=lambda x: x['CosineSimilarity'], reverse=True)

    # get_context = clist.context_list[s_bert_res_list[0]['context_idx']]
    get_context = []
    context_idx = []
    for i in range(3):
        print('比對的句子:', s_bert_res_list[i]['seg_context'])
        print('分數:', s_bert_res_list[i]['CosineSimilarity'])
        print('context:', clist.context_list[s_bert_res_list[i]['context_idx']])
        print('---')
        if s_bert_res_list[i]['context_idx'] not in context_idx:
            context_idx.append(s_bert_res_list[i]['context_idx'])
            get_context.append(clist.context_list[s_bert_res_list[i]['context_idx']])
    
    answer_list = []
    for i in range(0, len(get_context), 1):
        answer_list.append(answer_question([question], [get_context[i]], tokenizer, model, device))

    return answer_list
    # return None


def get_bm25_rank(corpus, question):
    bm25_model = bm25.BM25(corpus)
    # 逆文件頻率
    # average_idf = sum(map(lambda k: float(bm25_model.idf[k]), bm25_model.idf.keys())) / len(bm25_model.idf.keys())
    score = bm25_model.get_scores(question)

    return score

def use_BM25_get_doc(question, tokenizer, model, device):
    context_list = clist.context_list
    seg_context_list = []

    for idx in range(len(context_list)):
        seg_context_list.append(ps.seg_sentence(context_list[idx], ps.stopwords))

    context_list_score = get_bm25_rank(seg_context_list, ps.seg_sentence(question, ps.stopwords))
    ans_idx_list = heapq.nlargest(len(context_list_score), range(len(context_list_score)), context_list_score.__getitem__)[:3]
    for idx in range(len(ans_idx_list)):
        print(context_list[ans_idx_list[idx]])

    answer_list = []
    for i in range(0, len(ans_idx_list), 1):
        answer_list.append(answer_question([question], [context_list[ans_idx_list[i]]], tokenizer, model, device))

    return answer_list


def chatBotFlow(question, tokenizer, model, device, s_bert_model):
    # TODO: 完整比對 > 模糊比對 > BERT，邏輯需再修改
    ans = perfect_match(question)

    if ans == '':
        print('fuzzy')
        ans = get_cutomize_ans(question)
    
    if ans == '':
        print('bert')
        ans = compare_all_context(question, tokenizer, model, device, s_bert_model)

    return ans


def fuzzy_match(query, sentences, threshold=0.7):
    fcm = FuzzyChineseMatch(ngram_range=(3, 3), analyzer='stroke')
    fcm.fit(sentences)
    fcm.transform([query], n=1)

    most_similar_score = fcm.get_similarity_score().max()

    if most_similar_score >= threshold:
      return fcm.get_index()
    
    return None

def get_cutomize_ans(query):
    # 模糊比對
    q_list = clist.perfect_match_question_list;
    a_list = clist.perfect_match_answer_list;
    chinese_question = []
    for q in q_list:
      chinese_question.append(re.sub(chinese1, "", q))

    match_q = fuzzy_match(query, chinese_question, threshold=0.8)
    if match_q != None:
        print(q_list[match_q[0][0]])
        return [{
            'answer': a_list[q_list.index(q_list[match_q[0][0]])]
        }]
    return ''

def perfect_match(question):
    q_list = clist.perfect_match_question_list;
    a_list = clist.perfect_match_answer_list;
    if question in q_list: 
        return [{
            'answer': a_list[q_list.index(question)]
        }]
    
    return ''
