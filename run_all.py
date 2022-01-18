# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, train_AT, train_RDrop, train_RAT, init_network, test, predict
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True,default='Bert')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'Datasets/data_SST'  # dataset

    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(6)
    torch.manual_seed(6)
    torch.cuda.manual_seed_all(6)
    torch.backends.cudnn.deterministic = True  

    start_time = time.time()
    print("Loading data...",dataset)
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    
    # train
    print('----'*10) # ST
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)
    print('----'*10) # AT
    model = x.Model(config).to(config.device)
    train_AT(config, model, train_iter, dev_iter, test_iter)
    print('----'*10) # R-Drop
    model = x.Model(config).to(config.device)
    train_RDrop(config, model, train_iter, dev_iter, test_iter)
    print('----'*10) # R-AT
    model = x.Model(config).to(config.device)
    train_RAT(config, model, train_iter, dev_iter, test_iter)