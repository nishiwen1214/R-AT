# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer
CUDA_VISIBLE_DEVICES = 1
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Config(object):

    """config"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/train.csv'                                # train set
        self.dev_path = dataset + '/dev.csv'                                    # dev set
        self.test_path = dataset + '/dev.csv'                                  # test set
        self.class_list = [x.strip() for x in open(
            dataset + '/class.txt').readlines()]                                # classes
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.pt'        # save
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # gpu or cpu

        self.require_improvement = 10000                                 # early stopping
        self.num_classes = len(self.class_list)                         # number of classes
        self.num_epochs = 5                                             # epoch
        self.batch_size = 32                                          # mini-batch size
        self.pad_size = 70                                            # padding size
        self.learning_rate = 5e-5                                       # learning_rate
        self.bert_path = './bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.dropout = 0.3


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        context = x[0]  # input
        mask = x[2]  # mask
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        pooled = self.dropout(pooled)
        out = self.fc(pooled)
        return out
