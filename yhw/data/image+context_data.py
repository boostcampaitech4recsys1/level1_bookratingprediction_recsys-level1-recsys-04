## Data Loader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from tqdm import tqdm

def age_map(x: int) -> int:
    x = int(x)
    if x < 20:
        return 1
    elif x >= 20 and x < 30:
        return 2
    elif x >= 30 and x < 40:
        return 3
    elif x >= 40 and x < 50:
        return 4
    elif x >= 50 and x < 60:
        return 5
    else:
        return 6

def process_context_data(users, books, ratings1, ratings2):
    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
    users = users.drop(['location'], axis=1)

    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')

    # 인덱싱 처리
    loc_city2idx = {v:k for k,v in enumerate(context_df['location_city'].unique())}
    loc_state2idx = {v:k for k,v in enumerate(context_df['location_state'].unique())}
    loc_country2idx = {v:k for k,v in enumerate(context_df['location_country'].unique())}

    train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
    train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
    train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
    test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
    test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)

    train_df['age'] = train_df['age'].fillna(int(train_df['age'].mean()))
    train_df['age'] = train_df['age'].apply(age_map)
    test_df['age'] = test_df['age'].fillna(int(test_df['age'].mean()))
    test_df['age'] = test_df['age'].apply(age_map)

    # book 파트 인덱싱
    category2idx = {v:k for k,v in enumerate(context_df['category'].unique())}
    publisher2idx = {v:k for k,v in enumerate(context_df['publisher'].unique())}
    language2idx = {v:k for k,v in enumerate(context_df['language'].unique())}
    author2idx = {v:k for k,v in enumerate(context_df['book_author'].unique())}

    train_df['category'] = train_df['category'].map(category2idx)
    train_df['publisher'] = train_df['publisher'].map(publisher2idx)
    train_df['language'] = train_df['language'].map(language2idx)
    train_df['book_author'] = train_df['book_author'].map(author2idx)
    test_df['category'] = test_df['category'].map(category2idx)
    test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    test_df['language'] = test_df['language'].map(language2idx)
    test_df['book_author'] = test_df['book_author'].map(author2idx)

    idx = {
        "loc_city2idx":loc_city2idx,
        "loc_state2idx":loc_state2idx,
        "loc_country2idx":loc_country2idx,
        "category2idx":category2idx,
        "publisher2idx":publisher2idx,
        "language2idx":language2idx,
        "author2idx":author2idx,
    }

    return idx, train_df, test_df

class Image_Dataset(Dataset):
    def __init__(self, user_isbn_vector, img_vector, label):
        self.user_isbn_vector = user_isbn_vector
        self.img_vector = img_vector
        self.label = label
    def __len__(self):
        return self.user_isbn_vector.shape[0]
    def __getitem__(self, i):
        return {
                'user_isbn_vector' : torch.tensor(self.user_isbn_vector[i], dtype=torch.long),
                'img_vector' : torch.tensor(self.img_vector[i], dtype=torch.float32),
                'label' : torch.tensor(self.label[i], dtype=torch.float32),
                }


def image_vector(path):
    img = Image.open(path)
    scale = transforms.Resize((32, 32))
    tensor = transforms.ToTensor()
    img_fe = Variable(tensor(scale(img)))
    return img_fe


def process_img_data(df, books, user2idx, isbn2idx, train=False):
    books_ = books.copy()
    books_['isbn'] = books_['isbn'].map(isbn2idx)

    if train == True:
        df_ = df.copy()
    else:
        df_ = df.copy()
        df_['user_id'] = df_['user_id'].map(user2idx)
        df_['isbn'] = df_['isbn'].map(isbn2idx)

    df_ = pd.merge(df_, books_[['isbn', 'img_path']], on='isbn', how='left')
    df_['img_path'] = df_['img_path'].apply(lambda x: 'data/'+x)
    img_vector_df = df_[['img_path']].drop_duplicates().reset_index(drop=True).copy()
    data_box = []
    for idx, path in tqdm(enumerate(sorted(img_vector_df['img_path']))):
        data = image_vector(path)
        if data.size()[0] == 3:
            data_box.append(np.array(data))
        else:
            data_box.append(np.array(data.expand(3, data.size()[1], data.size()[2])))
    img_vector_df['img_vector'] = data_box
    df_ = pd.merge(df_, img_vector_df, on='img_path', how='left')
    return df_



def image_context_data_load(args):

    ######################## DATA LOAD
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
    test['user_id'] = test['user_id'].map(user2idx)
    users['user_id'] = users['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)
    books['isbn'] = books['isbn'].map(isbn2idx)

    idx, context_train, context_test = process_context_data(users, books, train, test)
    
    img_train = process_img_data(train, books, user2idx, isbn2idx, train=True)
    img_test = process_img_data(test, books, user2idx, isbn2idx, train=False)
    
    field_dims = np.array([len(user2idx), len(isbn2idx),
                            6, len(idx['loc_city2idx']), len(idx['loc_state2idx']), len(idx['loc_country2idx']),
                            len(idx['category2idx']), len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx'])], dtype=np.uint32)

    data = {
            'train':context_train,
            'test':context_test.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            'img_train':img_train,
            'img_test':img_test,
            }


    return data


def image_context_data_split(args, data):
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['rating'], axis=1),
                                                        data['train']['rating'],
                                                        test_size=args.TEST_SIZE,
                                                        random_state=args.SEED,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data

def image_context_data_loader(args, data):
    train_dataset = Image_Dataset(data['X_train'][['user_id', 'isbn','category','publisher','language','book_author']].values,
                                  data['X_train']['image_vector'].values,
                                  data['y_train'].values
                                ),
                                  
    valid_dataset = Image_Dataset(
                                data['X_valid'][['user_id', 'isbn','category','publisher','language','book_author']].values,
                                data['X_valid']['img_vector'].values,
                                data['y_valid'].values
                                )
    test_dataset = Image_Dataset(
                                data['img_test'][['user_id', 'isbn','category','publisher','language','book_author']].values,
                                data['img_test']['img_vector'].values,
                                data['img_test']['rating'].values
                                )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.BATCH_SIZE, num_workers=0, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, num_workers=0, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.BATCH_SIZE, num_workers=0, shuffle=False)
    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data
