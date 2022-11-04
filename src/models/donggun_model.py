import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ._models import _donggun
from ._models import rmse, RMSELoss
import wandb

class donggun:
    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()
        # self.criterion = nn.CrossEntropyLoss().to(args.DEVICE)


        self.train_dataloader = data['train_dataloader']
        self.valid_dataloader = data['valid_dataloader']
        self.field_dims = data['field_dims']

        self.embed_dim = args.FFM_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100

        self.device = args.DEVICE

        # for WDN
        # self.mlp_dims = args.WDN_MLP_DIMS
        # self.dropout = args.WDN_DROPOUT

        # for DCN
        self.mlp_dims = args.DCN_MLP_DIMS
        self.dropout = args.DCN_DROPOUT
        self.num_layers = args.DCN_NUM_LAYERS

        self.model = _donggun(self.field_dims, self.embed_dim, num_layers=self.num_layers, mlp_dims=self.mlp_dims, dropout=self.dropout).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)

    def train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        wandb.watch(self.model)
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.train_dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                fields, target = fields.to(self.device), target.to(self.device)

                y = self.model(fields)
                loss = self.criterion(y, target.float())
                # y = self.model(fields) #모델에서 소맥해서 꺼내고.
                # print(y.shape)
                # print(target.shape)
                # loss = self.criterion(y, target.long())
                # loss = self.criterion(y, target)

                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            rmse_score = self.predict_train()
            wandb.log({
                "RMSE": rmse_score,
                "Loss": total_loss
                })
            print('epoch:', epoch, 'validation: rmse:', rmse_score)


    def predict_train(self):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.valid_dataloader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        return rmse(targets, predicts)


    def predict(self, dataloader, data):
        self.model.eval()
        predicts = list()

        # self.data = data #일단 다 가져와 for cold start
        # no_exist = set(self.data['test']['user_id']) - set(self.data['train']['user_id'])
        #학습셋에는 없는 테스트셋 유저, 즉 콜드 스타트. 참고로 전처리 과정에서 이들은 고유 인덱스로 치환됐다.
        #이제 이 놈들한테 IBCF기반의 추천을 해줄 수 있으면 된다.

        with torch.no_grad():
            for fields in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
                #fields의 한 값은 유저id, isbn, 전처리 과정에 들어간 범주들 해서 10개의 차원인 리스트
                # if int(fields[0][0][0]) in no_exist:
                #     predicts.extend([5.8])
                #     # 콜드 스타트는 다 7점 때려보자
                #     continue
                fields = fields[0].to(self.device)



                y = self.model(fields)
                predicts.extend(y.tolist())
        return predicts
