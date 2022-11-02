import abc
import os
import re
from urllib.request import AbstractDigestAuthHandler
import nltk
from nltk import tokenize
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from transformers import BertModel, BertTokenizer


def text_preprocessing(summary): # text processing : 전처리 , re.sub 은 해당 문자열에 들어가있는 단어들을 빼준다. lower을 통해 다 소문자로 전처리!
    summary = re.sub("[.,\'\"''""!?]", "", summary)
    summary = re.sub("[^0-9a-zA-Z\\s]", " ", summary)
    summary = re.sub("\s+", " ", summary)
    summary = summary.lower()
    return summary

def summary_merge(df, user_id, max_summary): # summary 를 한 문장으로 하비기 특정 user_id 를 가지고 있는 row를 모아 길이 순서대로 정렬 한 뒤(길이가 긴 것 부터 짧은 것 순서대로) max_summary 갯수만큼 뽑은 뒤 한 문장으로 묶어 return
    return " ".join(df[df['user_id'] == user_id].sort_values(by='summary_length', ascending=False)['summary'].values[:max_summary])

def title_merge(df, user_id, max_summary): # summary 를 한 문장으로 하비기 특정 user_id 를 가지고 있는 row를 모아 길이 순서대로 정렬 한 뒤(길이가 긴 것 부터 짧은 것 순서대로) max_summary 갯수만큼 뽑은 뒤 한 문장으로 묶어 return
    print("!")
    return " ".join(df[df['user_id'] == user_id].sort_values(by='book_title_length', ascending=False)['summary'].values[:max_summary])


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


def process_text_data(df, books, user2idx, isbn2idx, device, train=False, user_summary_merge_vector=False, item_summary_vector=False):
    train = False
    books_ = books.copy()
    books_['isbn'] = books_['isbn'].map(isbn2idx) #isbn 을 idx로 바꿔줌

    if train == True:
        df_ = df.copy()
    else:
        df_ = df.copy()
        df_['user_id'] = df_['user_id'].map(user2idx) 
        df_['isbn'] = df_['isbn'].map(isbn2idx)

    df_ = pd.merge(df_, books_[['isbn', 'book_title','summary']], on='isbn', how='left') # 왼쪽 dataframe을 기준으로 join, 왼쪽 값 기준으로 없는 값은 nan으로 표시 , isbn을 기준으로 merge
    df_['summary'].fillna('None', inplace=True) # nan 처리
    df_['summary'] = df_['summary'].apply(lambda x:text_preprocessing(x)) # summary에 위에서 정의했던 text_proecssion을 처리
    df_['summary'].replace({'':'None', ' ':'None'}, inplace=True) # ' ' 이나 '' 처럼 빈칸으로 된 부분 대체
    df_['summary_length'] = df_['summary'].apply(lambda x:len(x)) # summary lengh 라는 column을 summary에 대한 length로 지정

    df_['book_title'].fillna('None', inplace=True) # na n 처리
    df_['book_title'] = df_['book_title'].apply(lambda x:text_preprocessing(x)) # summary에 위에서 정의했던 text_proecssion을 처리
    df_['book_title'].replace({'':'None', ' ':'None'}, inplace=True) # ' ' 이나 '' 처럼 빈칸으로 된 부분 대체
    df_['book_title_length'] = df_['book_title'].apply(lambda x:len(x)) # summary lengh 라는 column을 summary에 대한 length로 지정

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
        
        
        print('Create User Title Merge Vector')
        user_title_merge_vector_list = []
        for user in tqdm(df_['user_id'].unique()):
            vector = text_to_vector(title_merge(df_, user, 5), tokenizer, model, device) # 특정 user 에 대해서 긴 문장 위주로 합쳐서 merge해서 내보내줌, 그다음 vector로 변환
            user_title_merge_vector_list.append(vector)
        user_review_text_df = pd.DataFrame(df_['user_id'].unique(), columns=['user_id']) #
        user_review_text_df['user_title_merge_vector'] = user_title_merge_vector_list
        vector = np.concatenate([
                                user_review_text_df['user_id'].values.reshape(1, -1),
                                user_review_text_df['user_title_merge_vector'].values.reshape(1, -1)
                                ])
        if not os.path.exists('./data/text_vector'):
            os.makedirs('./data/text_vector')
        if train == True:
            np.save('./data/text_vector/train_user_title_merge_vector.npy', vector)
        else:
            np.save('./data/text_vector/test_user_title_merge_vector.npy', vector)
        
        
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
        
            
        print('Create Item Title Vector')
        item_title_vector_list = []
        books_title_df = df_[['isbn', 'book_title']].copy()
        books_title_df= books_text_df.drop_duplicates().reset_index(drop=True) # 중복 행 제거, index 새로 만들기
        books_title_df['book_title'].fillna('None', inplace=True) 
        for title in tqdm(books_text_df['book_title']):
            vector = text_to_vector(title, tokenizer, model, device) # summary에 대해서 tokenize
            item_title_vector_list.append(vector) 
        
        books_title_df['item_title_vector'] = item_title_vector_list
        vector = np.concatenate([
                                books_title_df['isbn'].values.reshape(1, -1),
                                books_title_df['item_title_vector'].values.reshape(1, -1)
                                ])
        if not os.path.exists('./data/text_vector'):
            os.makedirs('./data/text_vector')
        if train == True:
            np.save('./data/text_vector/train_item_title_vector.npy', vector)
        else:
            np.save('./data/text_vector/test_item_title_vector.npy', vector)    
            
    else:
        print('Check Vectorizer') # text vector 만드는 것 여부
        print('Vector Load')
        # user_summary
        if train == True:
            user = np.load('data/text_vector/train_user_summary_merge_vector.npy', allow_pickle=True)
        else:
            user = np.load('data/text_vector/test_user_summary_merge_vector.npy', allow_pickle=True)
        user_review_text_df = pd.DataFrame([user[0], user[1]]).T # user review
        user_review_text_df.columns = ['user_id', 'user_summary_merge_vector']
        user_review_text_df['user_id'] = user_review_text_df['user_id'].astype('int')
        
        # item_summary
        if train == True:
            item = np.load('data/text_vector/train_item_summary_vector.npy', allow_pickle=True)
        else:
            item = np.load('data/text_vector/test_item_summary_vector.npy', allow_pickle=True)
        books_text_df = pd.DataFrame([item[0], item[1]]).T
        books_text_df.columns = ['isbn', 'item_summary_vector']
        books_text_df['isbn'] = books_text_df['isbn'].astype('int')
        
        # user_title
        if train == True:
            user_title = np.load('data/text_vector/train_user_title_merge_vector.npy', allow_pickle=True)
        else:
            user_title = np.load('data/text_vector/test_user_title_merge_vector.npy', allow_pickle=True)
        user_title_df = pd.DataFrame([user_title[0], user_title[1]]).T
        user_title_df.columns = ['user_id', 'user_title_vector']
        user_title_df['user_id'] = user_title_df['user_id'].astype('int')
        
        # item_title
        if train == True:
            item_title = np.load('data/text_vector/train_item_title_vector.npy', allow_pickle=True)
        else:
            item_title = np.load('data/text_vector/test_item_title_vector.npy', allow_pickle=True)
        item_title_df = pd.DataFrame([item_title[0], item_title[1]]).T
        item_title_df.columns = ['isbn', 'item_title_vector']
        item_title_df['isbn'] = item_title_df['isbn'].astype('int')
        
        
        


    df_ = pd.merge(df_, user_review_text_df, on='user_id', how='left') #review 처리
    df_ = pd.merge(df_, user_title_df, on='user_id', how='left') #review 처리
    
    df_ = pd.merge(df_, books_text_df[['isbn', 'item_summary_vector']], on='isbn', how='left') # book 처리
    df_ = pd.merge(df_, item_title_df[['isbn', 'item_title_vector']], on='isbn', how='left') # book 처리

    return df_


class Text_Dataset(Dataset):
    def __init__(self, user_isbn_vector, user_summary_merge_vector, user_title_vector,
                 item_summary_vector, item_title_vector,label):
        self.user_isbn_vector = user_isbn_vector
        self.user_summary_merge_vector = user_summary_merge_vector
        self.item_summary_vector = item_summary_vector
        self.user_title_vector = user_title_vector
        self.item_title_vector = item_title_vector,
        self.label = label

    def __len__(self): 
        return self.user_isbn_vector.shape[0]

    def __getitem__(self, i):
        return {
                'user_isbn_vector' : torch.tensor(self.user_isbn_vector[i], dtype=torch.long),
                'user_summary_merge_vector' : torch.tensor(self.user_summary_merge_vector[i].reshape(-1, 1), dtype=torch.float32),
                'item_summary_vector' : torch.tensor(self.item_summary_vector[i].reshape(-1, 1), dtype=torch.float32),
                'item_title_vector' : torch.tensor(self.item_summary_vector[i].reshape(-1, 1), dtype = torch.float32),
                'user_title_vector' : torch.tensor(self.user_title_vector[i].reshape(-1, 1), dtype = torch.float32),
                'label' : torch.tensor(self.label[i], dtype=torch.float32),
                }

# option 에 따른 처리
def text_context_data_load(args):

    users = pd.read_csv(args.DATA_PATH + 'users.csv')
    books = pd.read_csv(args.DATA_PATH + 'books.csv')
    train = pd.read_csv(args.DATA_PATH + 'train_ratings.csv')
    test = pd.read_csv(args.DATA_PATH + 'test_ratings.csv')
    sub = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')

    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)

    text_train = process_text_data(train, books, user2idx, isbn2idx, args.DEVICE, train=True, user_summary_merge_vector=args.DEEPCONN_VECTOR_CREATE, item_summary_vector=args.DEEPCONN_VECTOR_CREATE)
    text_test = process_text_data(test, books, user2idx, isbn2idx, args.DEVICE, train=False, user_summary_merge_vector=args.DEEPCONN_VECTOR_CREATE, item_summary_vector=args.DEEPCONN_VECTOR_CREATE)

    data = {
            'train':train,
            'test':test,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            'text_train':text_train,
            'text_test':text_test,
            }

    return data


def text_context_data_split(args, data):
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['text_train'][['user_id', 'isbn', 'user_summary_merge_vector', 'item_summary_vector']],
                                                        data['text_train']['rating'],
                                                        test_size=args.TEST_SIZE,
                                                        random_state=args.SEED,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data


def text_context_data_loader(args, data):
    train_dataset = Text_Dataset(
                                data['X_train'][['user_id', 'isbn']].values,
                                data['X_train']['user_summary_merge_vector'].values,
                                data['X_train']['item_summary_vector'].values,
                                data['X_train']['item_title_vector'].values,
                                # data['X_train']['user_title_vector'].values,
                                data['y_train'].values
                                )
    valid_dataset = Text_Dataset(
                                data['X_valid'][['user_id', 'isbn']].values,
                                data['X_valid']['user_summary_merge_vector'].values,
                                data['X_valid']['item_summary_vector'].values,
                                data['X_valid']['item_title_vector'].values,
                                # data['X_valid']['user_title_vector'].values,
                                data['y_valid'].values
                                )
    test_dataset = Text_Dataset(
                                data['text_test'][['user_id', 'isbn']].values,
                                data['text_test']['user_summary_merge_vector'].values,
                                data['text_test']['item_summary_vector'].values,
                                data['text_test']['item_title_vector'].values,
                                # data['text_test']['user_title_vector'].values,
                                data['text_test']['rating'].values
                                )


    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.BATCH_SIZE, num_workers=0, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, num_workers=0, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.BATCH_SIZE, num_workers=0, shuffle=False)
    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data


# ## Data Loader
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset

# import os
# import re
# import nltk
# from nltk import tokenize

# from torch.autograd import Variable
# from tqdm import tqdm
# from torch.autograd import Variable
# from transformers import BertModel, BertTokenizer

# def age_map(x: int) -> int:
#     x = int(x)
#     if x < 20:
#         return 1
#     elif x >= 20 and x < 30:
#         return 2
#     elif x >= 30 and x < 40:
#         return 3
#     elif x >= 40 and x < 50:
#         return 4
#     elif x >= 50 and x < 60:
#         return 5
#     else:
#         return 6

# def process_context_data(users, books, ratings1, ratings2):
#     users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
#     users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
#     users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
#     users = users.drop(['location'], axis=1)

#     ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

#     # 인덱싱 처리된 데이터 조인
#     context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')
#     train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')
#     test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')

#     # 인덱싱 처리
#     loc_city2idx = {v:k for k,v in enumerate(context_df['location_city'].unique())}
#     loc_state2idx = {v:k for k,v in enumerate(context_df['location_state'].unique())}
#     loc_country2idx = {v:k for k,v in enumerate(context_df['location_country'].unique())}

#     train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
#     train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
#     train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
#     test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
#     test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
#     test_df['location_country'] = test_df['location_country'].map(loc_country2idx)

#     train_df['age'] = train_df['age'].fillna(int(train_df['age'].mean()))
#     train_df['age'] = train_df['age'].apply(age_map)
#     test_df['age'] = test_df['age'].fillna(int(test_df['age'].mean()))
#     test_df['age'] = test_df['age'].apply(age_map)

#     # book 파트 인덱싱
#     category2idx = {v:k for k,v in enumerate(context_df['category'].unique())}
#     publisher2idx = {v:k for k,v in enumerate(context_df['publisher'].unique())}
#     language2idx = {v:k for k,v in enumerate(context_df['language'].unique())}
#     author2idx = {v:k for k,v in enumerate(context_df['book_author'].unique())}

#     train_df['category'] = train_df['category'].map(category2idx)
#     train_df['publisher'] = train_df['publisher'].map(publisher2idx)
#     train_df['language'] = train_df['language'].map(language2idx)
#     train_df['book_author'] = train_df['book_author'].map(author2idx)
#     test_df['category'] = test_df['category'].map(category2idx)
#     test_df['publisher'] = test_df['publisher'].map(publisher2idx)
#     test_df['language'] = test_df['language'].map(language2idx)
#     test_df['book_author'] = test_df['book_author'].map(author2idx)

#     idx = {
#         "loc_city2idx":loc_city2idx,
#         "loc_state2idx":loc_state2idx,
#         "loc_country2idx":loc_country2idx,
#         "category2idx":category2idx,
#         "publisher2idx":publisher2idx,
#         "language2idx":language2idx,
#         "author2idx":author2idx,
#     }

#     return idx, train_df, test_df

# def text_preprocessing(summary): # text processing : 전처리 , re.sub 은 해당 문자열에 들어가있는 단어들을 빼준다. lower을 통해 다 소문자로 전처리!
#     summary = re.sub("[.,\'\"''""!?]", "", summary)
#     summary = re.sub("[^0-9a-zA-Z\\s]", " ", summary)
#     summary = re.sub("\s+", " ", summary)
#     summary = summary.lower()
#     return summary


# def summary_merge(df, user_id, max_summary): # summary 를 한 문장으로 합치고 특정 user_id 를 가지고 있는 row를 모아 길이 순서대로 정렬 한 뒤(길이가 긴 것 부터 짧은 것 순서대로) max_summary 갯수만큼 뽑은 뒤 한 문장으로 묶어 return
#     return " ".join(df[df['user_id'] == user_id].sort_values(by='summary_length', ascending=False)['summary'].values[:max_summary])


# def text_to_vector(text, tokenizer, model, device): 
#     for sent in tokenize.sent_tokenize(text): # 문장 단위 토크나이즈
#         text_ = "[CLS] " + sent + " [SEP]" # 먼저 Token Embedding에서는 두 가지 특수 토큰(CLS, SEP)을 사용하여 문장을 구별하게 되는데요. Special Classification token(CLS)은 모든 문장의 가장 첫 번째(문장의 시작) 토큰으로 삽입됩니다. 이 토큰은 Classification task에서는 사용되지만, 그렇지 않을 경우엔 무시됩니다. 
#                                             #또, Special Separator token(SEP)을 사용하여 첫 번째 문장과 두 번째 문장을 구별합니다. 여기에 segment Embedding을 더해서 앞뒤 문장을 더욱 쉽게 구별할 수 있도록 도와줍니다. 이 토큰은 각 문장의 끝에 삽입됩니다.
#         tokenized = tokenizer.tokenize(text_) 
#         indexed = tokenizer.convert_tokens_to_ids(tokenized) # BERT 를 위한 token, sgments tensor 만들기
#         segments_idx = [1] * len(tokenized) 
#         token_tensor = torch.tensor([indexed])
#         sgments_tensor = torch.tensor([segments_idx]) 
#         with torch.no_grad():
#             outputs = model(token_tensor.to(device), sgments_tensor.to(device))
#             encode_layers = outputs[0]
#             sentence_embedding = torch.mean(encode_layers[0], dim=0)
#     return sentence_embedding.cpu().detach().numpy()


# def process_text_data(df, books, user2idx, isbn2idx, device, train=False, user_summary_merge_vector=False, item_summary_vector=False, item_title_vector=False):
#     books_ = books.copy()
#     books_['isbn'] = books_['isbn'].map(isbn2idx) #isbn 을 idx로 바꿔줌

#     if train == True:
#         df_ = df.copy()
#     else:
#         df_ = df.copy()
#         df_['user_id'] = df_['user_id'].map(user2idx) 
#         df_['isbn'] = df_['isbn'].map(isbn2idx)
    
#     df_ = pd.merge(df_, books_[[ 'isbn','book_title','summary']], on='isbn', how='left') # 왼쪽 dataframe을 기준으로 join, 왼쪽 값 기준으로 없는 값은 nan으로 표시 , isbn을 기준으로 merge
#     df_['summary'].fillna('None', inplace=True) # na n 처리
#     df_['summary'] = df_['summary'].apply(lambda x:text_preprocessing(x)) # summary에 위에서 정의했던 text_proecssion을 처리
#     df_['summary'].replace({'':'None', ' ':'None'}, inplace=True) # ' ' 이나 '' 처럼 빈칸으로 된 부분 대체
#     df_['summary_length'] = df_['summary'].apply(lambda x:len(x)) # summary lengh 라는 column을 summary에 대한 length로 지정

#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')   # bert tokenizer model 불러옴 lower case로 다 바꿔서 lower case에 관한 tokenizer만 불러옴
#     model = BertModel.from_pretrained('bert-base-uncased').to(device) # pretrained bert model 불러옴
#     print('train :'+str(train))
#     if user_summary_merge_vector and item_summary_vector:  
#         print('Create User Summary Merge Vector')
#         user_summary_merge_vector_list = []
#         for user in tqdm(df_['user_id'].unique()):
#             vector = text_to_vector(summary_merge(df_, user, 5), tokenizer, model, device) # 특정 user 에 대해서 긴 문장 위주로 합쳐서 merge해서 내보내줌, 그다음 vector로 변환
#             user_summary_merge_vector_list.append(vector) 
#         user_review_text_df = pd.DataFrame(df_['user_id'].unique(), columns=['user_id']) #
#         user_review_text_df['user_summary_merge_vector'] = user_summary_merge_vector_list
#         vector = np.concatenate([
#                                 user_review_text_df['user_id'].values.reshape(1, -1),
#                                 user_review_text_df['user_summary_merge_vector'].values.reshape(1, -1)
#                                 ])
#         if not os.path.exists('./data/text_vector'):
#             os.makedirs('./data/text_vector')
#         if train == True: 
#             np.save('./data/text_vector/train_user_summary_merge_vector.npy', vector)
#         else:
#             np.save('./data/text_vector/test_user_summary_merge_vector.npy', vector)

#         print('Create Item Summary Vector')
#         item_summary_vector_list = []
#         books_text_df = df_[['isbn', 'summary']].copy()
#         books_text_df= books_text_df.drop_duplicates().reset_index(drop=True) # 중복 행 제거, index 새로 만들기
#         books_text_df['summary'].fillna('None', inplace=True) 
#         for summary in tqdm(books_text_df['summary']):
#             vector = text_to_vector(summary, tokenizer, model, device) # summary에 대해서 tokenize
#             item_summary_vector_list.append(vector) 
#         books_text_df['item_summary_vector'] = item_summary_vector_list
#         vector = np.concatenate([
#                                 books_text_df['isbn'].values.reshape(1, -1),
#                                 books_text_df['item_summary_vector'].values.reshape(1, -1)
#                                 ]) # 합침
#         if not os.path.exists('./data/text_vector'):
#             os.makedirs('./data/text_vector')
#         if train == True:
#             np.save('./data/text_vector/train_item_summary_vector.npy', vector)
#         else:
#             np.save('./data/text_vector/test_item_summary_vector.npy', vector)
        
        
#         print('Create Item Title Vector')
#         item_title_vector_list = []
#         books_title_df = df_[['isbn', 'book_title']].copy()
#         books_title_df= books_text_df.drop_duplicates().reset_index(drop=True) # 중복 행 제거, index 새로 만들기
#         books_title_df['book_title'].fillna('None', inplace=True) 
#         for title in tqdm(books_text_df['book_title']):
#             vector = text_to_vector(title, tokenizer, model, device) # summary에 대해서 tokenize
#             item_title_vector_list.append(vector) 
        
#         books_title_df['item_title_vector'] = item_title_vector_list
#         vector = np.concatenate([
#                                 books_title_df['isbn'].values.reshape(1, -1),
#                                 books_title_df['item_title_vector'].values.reshape(1, -1)
#                                 ])
#         if not os.path.exists('./data/text_vector'):
#             os.makedirs('./data/text_vector')
#         if train == True:
#             np.save('./data/text_vector/train_item_title_vector.npy', vector)
#         else:
#             np.save('./data/text_vector/test_item_title_vector.npy', vector)
        
#     else:
#         print('Check Vectorizer') # text vector 만드는 것 여부
#         print('Vector Load')
#         if train == True:
#             user = np.load('data/text_vector/train_user_summary_merge_vector.npy', allow_pickle=True)
#         else:
#             user = np.load('data/text_vector/test_user_summary_merge_vector.npy', allow_pickle=True)
#         user_review_text_df = pd.DataFrame([user[0], user[1]]).T # user review
#         user_review_text_df.columns = ['user_id', 'user_summary_merge_vector']
#         user_review_text_df['user_id'] = user_review_text_df['user_id'].astype('int')

#         if train == True:
#             item = np.load('data/text_vector/train_item_summary_vector.npy', allow_pickle=True)
#         else:
#             item = np.load('data/text_vector/test_item_summary_vector.npy', allow_pickle=True)
#         books_text_df = pd.DataFrame([item[0], item[1]]).T
#         books_text_df.columns = ['isbn', 'item_summary_vector']
#         books_text_df['isbn'] = books_text_df['isbn'].astype('int')
        
#         if train == True:
#             item = np.load('data/text_vector/train_item_title_vector.npy', allow_pickle=True)
#         else:
#             item = np.load('data/text_vector/test_item_title_vector.npy', allow_pickle=True)
#         books_title_df = pd.DataFrame([item[0], item[1]]).T
#         books_title_df.columns = ['isbn', 'item_title_vector']
#         books_title_df['isbn'] = books_title_df['isbn'].astype('int')


#     df_ = pd.merge(df_, user_review_text_df, on='user_id', how='left') #review 처리
#     df_ = pd.merge(df_, books_text_df[['isbn', 'item_summary_vector']], on='isbn', how='left') # book 처리
#     df_ = pd.merge(df_, books_title_df[['isbn', 'item_title_vector']], on='isbn', how='left') # book 처리

#     return df_


# class Text_Dataset(Dataset):
#     def __init__(self, user_isbn_vector, user_summary_merge_vector, item_summary_vector, item_title_vector,label):
#         self.user_isbn_vector = user_isbn_vector
#         self.user_summary_merge_vector = user_summary_merge_vector
#         self.item_summary_vector = item_summary_vector
#         self.item_title_vector = item_title_vector,
#         self.label = label

#     def __len__(self): 
#         return self.user_isbn_vector.shape[0]

#     def __getitem__(self, i):
#         return {
#                 'user_isbn_vector' : torch.tensor(self.user_isbn_vector[i], dtype=torch.long),
#                 'user_summary_merge_vector' : torch.tensor(self.user_summary_merge_vector[i].reshape(-1, 1), dtype=torch.float32),
#                 'item_summary_vector' : torch.tensor(self.item_summary_vector[i].reshape(-1, 1), dtype=torch.float32),
#                 'item_title_vector' : torch.tensor(self.item_title_vector[i].reshape(-1, 1), dtype=torch.float32),
#                 'label' : torch.tensor(self.label[i], dtype=torch.float32)
#                 }


# def text_context_data_load(args):

#     users = pd.read_csv(args.DATA_PATH + 'users.csv')
#     books = pd.read_csv(args.DATA_PATH + 'books.csv')
#     train = pd.read_csv(args.DATA_PATH + 'train_ratings.csv')
#     test = pd.read_csv(args.DATA_PATH + 'test_ratings.csv')
#     sub = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')

#     ids = pd.concat([train['user_id'], sub['user_id']]).unique()
#     isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

#     idx2user = {idx:id for idx, id in enumerate(ids)}
#     idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

#     user2idx = {id:idx for idx, id in idx2user.items()}
#     isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

#     train['user_id'] = train['user_id'].map(user2idx)
#     sub['user_id'] = sub['user_id'].map(user2idx)

#     train['isbn'] = train['isbn'].map(isbn2idx)
#     sub['isbn'] = sub['isbn'].map(isbn2idx)

#     text_train = process_text_data(train, books, user2idx, isbn2idx, args.DEVICE, train=True, user_summary_merge_vector=args.DEEPCONN_VECTOR_CREATE, item_summary_vector=args.DEEPCONN_VECTOR_CREATE, item_title_vector=args.DEEPCONN_VECTOR_CREATE)
#     text_test = process_text_data(test, books, user2idx, isbn2idx, args.DEVICE, train=False, user_summary_merge_vector=args.DEEPCONN_VECTOR_CREATE, item_summary_vector=args.DEEPCONN_VECTOR_CREATE, item_title_vector=args.DEEPCONN_VECTOR_CREATE)

#     data = {
#             'train':train,
#             'test':test,
#             'users':users,
#             'books':books,
#             'sub':sub,
#             'idx2user':idx2user,
#             'idx2isbn':idx2isbn,
#             'user2idx':user2idx,
#             'isbn2idx':isbn2idx,
#             'text_train':text_train,
#             'text_test':text_test,
#             }


#     return data


# def text_context_data_split(args, data):
#     X_train, X_valid, y_train, y_valid = train_test_split(
#                                                         data['text_train'][['user_id', 'isbn','user_summary_merge_vector', 'item_summary_vector','item_title_vector']],
#                                                         data['text_train']['rating'],
#                                                         test_size=args.TEST_SIZE,
#                                                         random_state=args.SEED,
#                                                         shuffle=True
#                                                         )
#     data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
#     return data

# def text_context_data_loader(args, data):
#     train_dataset = Text_Dataset(
#                                 data['X_train'][['user_id', 'isbn',]].values,
#                                 data['X_train']['user_summary_merge_vector'].values,
#                                 data['X_train']['item_summary_vector'].values,
#                                 data['X_train']['item_title_vector'].values,
#                                 data['y_train'].values
#                                 ),
                                  
#     valid_dataset = Text_Dataset(
#                                 data['X_valid'][['user_id', 'isbn',]].values,
#                                 data['X_valid']['user_summary_merge_vector'].values,
#                                 data['X_valid']['item_summary_vector'].values,
#                                 data['X_valid']['item_title_vector'].values,
#                                 data['X_valid'].values
#                                 )
#     test_dataset = Text_Dataset(
#                                 data['text_test'][['user_id', 'isbn',]].values,
#                                 data['text_test']['user_summary_merge_vector'].values,
#                                 data['text_test']['item_summary_vector'].values,
#                                 data['text_test']['item_title_vector'].values,
#                                 data['text_test'].values
#                                 )

#     train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.BATCH_SIZE, num_workers=0, shuffle=True)
#     valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, num_workers=0, shuffle=True)
#     test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.BATCH_SIZE, num_workers=0, shuffle=False)
#     data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

#     return data
