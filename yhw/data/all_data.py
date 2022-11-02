## Data Loader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import os
import re
import nltk
from nltk import tokenize

from torch.autograd import Variable
from tqdm import tqdm
from torch.autograd import Variable
from transformers import BertModel, BertTokenizer


# text 
def text_preprocessing(summary): # text processing : 전처리 , re.sub 은 해당 문자열에 들어가있는 단어들을 빼준다. lower을 통해 다 소문자로 전처리!
    summary = re.sub("[.,\'\"''""!?]", "", summary)
    summary = re.sub("[^0-9a-zA-Z\\s]", " ", summary)
    summary = re.sub("\s+", " ", summary)
    summary = summary.lower()
    return summary

def summary_merge(df, user_id, max_summary): # summary 를 한 문장으로 합치고 특정 user_id 를 가지고 있는 row를 모아 길이 순서대로 정렬 한 뒤(길이가 긴 것 부터 짧은 것 순서대로) max_summary 갯수만큼 뽑은 뒤 한 문장으로 묶어 return
    return " ".join(df[df['user_id'] == user_id].sort_values(by='summary_length', ascending=False)['summary'].values[:max_summary])


def text_to_vector(text, tokenizer, model, device): 
    for sent in tokenize.sent_tokenize(text): # 문장 단위 토크나이즈
        text_ = "[CLS] " + sent + " [SEP]" # 먼저 Token Embedding에서는 두 가지 특수 토큰(CLS, SEP)을 사용하여 문장을 구별하게 되는데요. Special Classification token(CLS)은 모든 문장의 가장 첫 번째(문장의 시작) 토큰으로 삽입됩니다. 이 토큰은 Classification task에서는 사용되지만, 그렇지 않을 경우엔 무시됩니다. 
                                            #또, Special Separator token(SEP)을 사용하여 첫 번째 문장과 두 번째 문장을 구별합니다. 여기에 segment Embedding을 더해서 앞뒤 문장을 더욱 쉽게 구별할 수 있도록 도와줍니다. 이 토큰은 각 문장의 끝에 삽입됩니다.
        tokenized = tokenizer.tokenize(text_) 
        indexed = tokenizer.convert_tokens_to_ids(tokenized) # BERT 를 위한 token, sgments tensor 만들기
        segments_idx = [1] * len(tokenized) 
        token_tensor = torch.tensor([indexed])
        sgments_tensor = torch.tensor([segments_idx]) 
        with torch.no_grad():
            outputs = model(token_tensor.to(device), sgments_tensor.to(device))
            encode_layers = outputs[0]
            sentence_embedding = torch.mean(encode_layers[0], dim=0)
    return sentence_embedding.cpu().detach().numpy()



# image




def process_text_data(df, books, user2idx, isbn2idx, device, train=False, user_summary_merge_vector=False, item_summary_vector=False):
    books_ = books.copy()
    books_['isbn'] = books_['isbn'].map(isbn2idx) #isbn 을 idx로 바꿔줌

    if train == True:
        df_ = df.copy()
    else:
        df_ = df.copy()
        df_['user_id'] = df_['user_id'].map(user2idx) 
        df_['isbn'] = df_['isbn'].map(isbn2idx)

    df_ = pd.merge(df_, books_[['isbn', 'summary']], on='isbn', how='left') # 왼쪽 dataframe을 기준으로 join, 왼쪽 값 기준으로 없는 값은 nan으로 표시 , isbn을 기준으로 merge
    df_['summary'].fillna('None', inplace=True) # nan 처리
    df_['summary'] = df_['summary'].apply(lambda x:text_preprocessing(x)) # summary에 위에서 정의했던 text_proecssion을 처리
    df_['summary'].replace({'':'None', ' ':'None'}, inplace=True) # ' ' 이나 '' 처럼 빈칸으로 된 부분 대체
    df_['summary_length'] = df_['summary'].apply(lambda x:len(x)) # summary lengh 라는 column을 summary에 대한 length로 지정

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')   # bert tokenizer model 불러옴 lower case로 다 바꿔서 lower case에 관한 tokenizer만 불러옴
    model = BertModel.from_pretrained('bert-base-uncased').to(device) # pretrained bert model 불러옴

    if user_summary_merge_vector and item_summary_vector:  
        print('Create User Summary Merge Vector')
        user_summary_merge_vector_list = []
        for user in tqdm(df_['user_id'].unique()):
            vector = text_to_vector(summary_merge(df_, user, 5), tokenizer, model, device) # 특정 user 에 대해서 긴 문장 위주로 합쳐서 merge해서 내보내줌, 그다음 vector로 변환
            user_summary_merge_vector_list.append(vector) 
        user_review_text_df = pd.DataFrame(df_['user_id'].unique(), columns=['user_id']) #
        user_review_text_df['user_summary_merge_vector'] = user_summary_merge_vector_list
        vector = np.concatenate([
                                user_review_text_df['user_id'].values.reshape(1, -1),
                                user_review_text_df['user_summary_merge_vector'].values.reshape(1, -1)
                                ])
        if not os.path.exists('./data/text_vector'):
            os.makedirs('./data/text_vector')
        if train == True:
            np.save('./data/text_vector/train_user_summary_merge_vector.npy', vector)
        else:
            np.save('./data/text_vector/test_user_summary_merge_vector.npy', vector)

        print('Create Item Summary Vector')
        item_summary_vector_list = []
        books_text_df = df_[['isbn', 'summary']].copy()
        books_text_df= books_text_df.drop_duplicates().reset_index(drop=True) # 중복 행 제거, index 새로 만들기
        books_text_df['summary'].fillna('None', inplace=True) 
        for summary in tqdm(books_text_df['summary']):
            vector = text_to_vector(summary, tokenizer, model, device) # summary에 대해서 tokenize
            item_summary_vector_list.append(vector) 
        books_text_df['item_summary_vector'] = item_summary_vector_list
        vector = np.concatenate([
                                books_text_df['isbn'].values.reshape(1, -1),
                                books_text_df['item_summary_vector'].values.reshape(1, -1)
                                ]) # 합침
        if not os.path.exists('./data/text_vector'):
            os.makedirs('./data/text_vector')
        if train == True:
            np.save('./data/text_vector/train_item_summary_vector.npy', vector)
        else:
            np.save('./data/text_vector/test_item_summary_vector.npy', vector)
    else:
        print('Check Vectorizer') # text vector 만드는 것 여부
        print('Vector Load')
        if train == True:
            user = np.load('data/text_vector/train_user_summary_merge_vector.npy', allow_pickle=True)
        else:
            user = np.load('data/text_vector/test_user_summary_merge_vector.npy', allow_pickle=True)
        user_review_text_df = pd.DataFrame([user[0], user[1]]).T # user review
        user_review_text_df.columns = ['user_id', 'user_summary_merge_vector']
        user_review_text_df['user_id'] = user_review_text_df['user_id'].astype('int')

        if train == True:
            item = np.load('data/text_vector/train_item_summary_vector.npy', allow_pickle=True)
        else:
            item = np.load('data/text_vector/test_item_summary_vector.npy', allow_pickle=True)
        books_text_df = pd.DataFrame([item[0], item[1]]).T
        books_text_df.columns = ['isbn', 'item_summary_vector']
        books_text_df['isbn'] = books_text_df['isbn'].astype('int')


    df_ = pd.merge(df_, user_review_text_df, on='user_id', how='left') #review 처리
    df_ = pd.merge(df_, books_text_df[['isbn', 'item_summary_vector']], on='isbn', how='left') # book 처리

    return df_