import pandas as pd
from sklearn.model_selection import train_test_split
import json
import argparse

class jsonEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        else:
            return super(jsonEncoder, self).default(obj)

def get_args():
    parser = argparse.ArgumentParser(description='Devide data into train data and test data')
    parser.add_argument('--path', 
                        help="The path of xlsx data",
                        required=True,
                        type=str)
    parser.add_argument('--prefix', 
                        help="Prefix string in train data name and test data name",
                        default= 'hcp',
                        type=str)
    parser.add_argument('--test_size', 
                        help="This parameter is same as sklearn's parameter. Its type should be float so far",
                        default= 0.1,
                        type=float)
                        

    args = parser.parse_args()
    
    return args

def read_excel_data(path: str):
    exel = pd.read_excel(path)
    exel = exel.assign(
            context = lambda df: df.context.str.replace(pat=' ', repl = ''),
            question = lambda df: df.question.str.replace(pat=' ', repl = ''),
            answer = lambda df: df.answer.str.replace(pat=' ', repl = '')
        ).assign(
            answer_start = lambda df: df.apply(
                axis = 1, 
                func = lambda row: 
                    row.context.find(row.answer)
            )
        )
    
    return exel

def write_to_squad_format(df_excel, output_data_name):
    data = []
    for i, context in enumerate(df_excel.context.unique()):
        qas = []
        sub_exel = df_excel.loc[lambda df: df.context == context]
        for r in sub_exel.index:
            q = sub_exel.question[r]
            a = sub_exel.answer[r]
            s = sub_exel.answer_start[r]
            qas.append(
                {
                    "question" : q,
                    "id" : f"TRAIN_{i}_QUERY_{r}",
                    "answers" : [
                        {
                            "text" : a,
                            "answer_start" : s
                        }
                    ]
                }
            )
        data.append(
            {
                "paragraphs" :[
                    {
                        "id" : f"TRAIN_{i}",
                        "context" : context,
                        "qas" : qas
                    }
                ],
                "id" : f"TRAIN_{i}",
                "title": f"source{i}"

            }

        )
        
    train_json = json.dumps({
        "version": "v1.0",
        "data" : data
    }, cls= jsonEncoder, ensure_ascii=False, indent=4)

    with open(output_data_name, 'w') as f:
        f.write(train_json)
        f.close()


def split_dataframe_to_squad_format(df_excel, train_data_name, test_data_name, test_size=0.1):
    train_exel, test_exel = train_test_split(df_excel, test_size=test_size, random_state=777)
    write_to_squad_format(train_exel, train_data_name)
    write_to_squad_format(test_exel, test_data_name)


if __name__ == '__main__':
    arg_list = get_args()
    excel = read_excel_data(arg_list.path)
    split_dataframe_to_squad_format(excel, 
                arg_list.prefix + '_train.json', 
                arg_list.prefix + '_test.json', 
                test_size=arg_list.test_size)
    
    print('Finish!')


