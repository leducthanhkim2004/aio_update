import os
import sys
import glob
import scipy
import pickle
import numpy as np
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
grand_dir = os.path.abspath(os.path.join(parent_dir, '..'))
sys.path.extend([parent_dir, grand_dir])
from utils.combine_utils import merge_searching_results_by_addition

def GET_PROJECT_ROOT():
    # goto the root folder of LogBar
    current_abspath = os.path.abspath(__file__)
    while True:
        if os.path.split(current_abspath)[1] == 'AIHCM':
            project_root = current_abspath
            break
        else:
            current_abspath = os.path.dirname(current_abspath)
    return project_root

PROJECT_ROOT = GET_PROJECT_ROOT()

class load_file:
    def __init__(
            self,
            clean_data_path,  # clean_data_path and context can't not be None at the same time
            save_tfids_object_path,
            update:bool,
            all_datatpye,
            context_data = None,
            ngram_range = (1, 1),
            input_datatype = 'txt',
    ):
        tfidf_transform = {}
        context_matrix = {}
        for data_type in all_datatpye:
            if (not os.path.exists(os.path.join(save_tfids_object_path, f'tfidf_transform_{data_type}.pkl'))):
                if context_data == None:
                    clean_data_paths = os.path.join(PROJECT_ROOT, clean_data_path[data_type])
                    context = self.load_context(clean_data_paths, input_datatype)
                    print(data_type)
                    print(context[0][:100])
                else:
                    context = context_data
                tfidf_transform[data_type] = TfidfVectorizer(input = 'content', ngram_range = ngram_range, token_pattern=r"(?u)\b[\w\d]+\b")
                context_matrix[data_type] = tfidf_transform[data_type].fit_transform(context).tocsr()
                print(tfidf_transform[data_type].get_feature_names_out()[:10])
                print(context_matrix[data_type].shape)
                with open(os.path.join(save_tfids_object_path, f'tfidf_transform_{data_type}.pkl'), 'wb') as f:
                    pickle.dump(tfidf_transform[data_type], f)
                scipy.sparse.save_npz(os.path.join(save_tfids_object_path, f'sparse_context_matrix_{data_type}.npz'), context_matrix[data_type])

    def load_context(self, clean_data_paths, input_datatype):
        context = []
        if input_datatype == 'txt':
            data_paths = []
            cxx_data_paths = glob.glob(clean_data_paths)
            cxx_data_paths.sort()
            for cxx_data_path in cxx_data_paths:
                data_path = glob.glob(cxx_data_path + '/*.txt')
                data_path.sort(reverse=False, key=lambda s:int(s[-7:-4]))
                data_paths += data_path
            for path in data_paths:
                with open(path, 'r', encoding='utf-8') as f:
                    data = f.readlines()
                    data = [item.strip() for item in data]
                    context += data
        elif input_datatype == 'json':
            context_paths = glob.glob(clean_data_paths)
            context_paths.sort()
            for cxx_context_path in context_paths:
                paths = glob.glob(cxx_context_path + '/*.json')
                paths.sort(reverse=False, key=lambda x: int(x[-8:-5]))
                for path in paths:
                    with open(path) as f:
                        context += [self.preprocess_text(' '.join(line)) for line in json.load(f)]
        else:
            print(f'not support reading the {input_datatype}')
            sys.exit()
        return context
    
    @staticmethod
    def preprocess_text(text:str):
        text = text.lower()
        # keep letter and number remove all remain
        reg_pattern = '[^a-z0-9A-Z_ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễếệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ\s]'
        output = re.sub(reg_pattern, '', text)
        output = output.strip()
        return output

if __name__ == '__main__':
    inputs = {
        'bbox': "a0kite b0kite",
        'class': "people1 tv1",
        'color':None,
        'tag':None,
        'number':None,
    }
    # obj = object_retrieval()
    #list_answer = obj(inputs, k=3)
    #print(list_answer)
    # obj.transform_input('query', 'input_type') # input_type is bbox, color, class, tag
    # context_vector = obj.get_context_vector() # get context vector
    pass