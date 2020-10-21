#!/usr/bin/env python
#created by Kanto

import argparse
import json
import logging
import os

import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import torch.utils.data
from torchvision import datasets, transforms
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import pandas as pd
import json
import collections
import numpy as np
import boto3
from sklearn.metrics import mean_squared_error as mse
from six import BytesIO

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def _average_gradients(model):
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        #dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0) old pytorch version
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


class Encoder(nn.Module):

    def __init__(self, batch_size, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.batch_size = batch_size
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.rnn1 = nn.LSTM(
          input_size=n_features,
          hidden_size=self.hidden_dim,
          num_layers=1,
          batch_first=True
        )

        self.rnn2 = nn.LSTM(
          input_size=self.hidden_dim,
          hidden_size=embedding_dim,
          num_layers=1,
          batch_first=True
        )

#you have to change batch_size as per your arg.batch_size

    def forward(self, x):
        
        '''
        print("beginning of the forward: ", x.size())
        print("batch size is: ",self.batch_size)
        print("seq_len size is: ",self.seq_len)
        print("n_features size is: ",self.n_features)
        '''

        x = x.reshape((self.batch_size, self.seq_len, self.n_features))

        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)

        return hidden_n.reshape((self.batch_size, self.n_features, self.embedding_dim))
    
class Decoder(nn.Module):

    def __init__(self, batch_size, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.batch_size = batch_size

        self.rnn1 = nn.LSTM(
          input_size=input_dim,
          hidden_size=input_dim,
          num_layers=1,
          batch_first=True
        )

        self.rnn2 = nn.LSTM(
          input_size=input_dim,
          hidden_size=self.hidden_dim,
          num_layers=1,
          batch_first=True
        )

        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        '''
        print("beginning of the forward: ", x.size())
        print("batch size is: ",self.batch_size)
        print("seq_len size is: ",self.seq_len)
        print("n_features size is: ",self.n_features)
        '''
        x = x.repeat(1, self.seq_len, self.n_features)
        #print(x.size())
        #x = x.reshape((self.batch_size, self.n_features, self.seq_len, self.input_dim))

        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((self.batch_size, self.seq_len, self.hidden_dim))

        return self.output_layer(x)
    
class RecurrentAutoencoder(nn.Module):

    def __init__(self, batch_size, seq_len, n_features, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()

        self.encoder = Encoder(batch_size, seq_len, n_features, embedding_dim)
        self.decoder = Decoder(batch_size, seq_len, embedding_dim, n_features)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
  
    
def data_loader(batch_size, training_dir, is_distributed, **kwargs):
    logger.info("Get train data loader")
    logger.info(training_dir)
    data = pd.read_csv(os.path.join(training_dir, 'donghwa-dscrn-shingle.csv')).values.astype(np.float32)
    #data = scaler.fit_transform(data).astype(np.float32)
    #train_sampler = torch.utils.data.distributed.DistributedSampler(data) if is_distributed else None
    #return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=train_sampler is None,
    #                                   sampler=train_sampler, **kwargs)
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle= False,drop_last=True)
    
    
def train(args):
    use_cuda = args.num_gpus > 0
    device = torch.device("cuda" if use_cuda else "cpu")
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    #device = torch.device("cuda" if use_cuda else "cpu")
    label = args.label
    
    seq_len = 30
    n_features = 1
    
    #(batchsize, seq_len, n_features)
    #
    
    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ['RANK'] = str(host_rank)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        logger.info('Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
            args.backend, dist.get_world_size()) + 'Current host rank is {}. Number of gpus: {}'.format(
            dist.get_rank(), args.num_gpus))

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    train_loader = data_loader(args.batch_size, args.data_dir, is_distributed, **kwargs)

    logger.debug("Processes {}/{} ({:.0f}%) of train data".format(
        len(train_loader.sampler), len(train_loader.dataset),
        100. * len(train_loader.sampler) / len(train_loader.dataset)
    ))
    model = RecurrentAutoencoder(args.batch_size, seq_len, n_features, 64)
    model = model.to(device)

    if is_distributed and True:
        # multi-machine multi-gpu case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = torch.nn.DataParallel(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, data in enumerate(train_loader, 1):
            #print("my first data shape is: ",data.size() )
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.mse_loss(data , output.squeeze())
            loss.backward()
            if is_distributed and not use_cuda:
                # average gradients manually for multi-machine cpu case only
                _average_gradients(model)
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.sampler),
                    100. * batch_idx / len(train_loader), loss.item()))
    save_model(model, args.model_dir)
    
def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    torch.save(model.cpu().state_dict(), path)
    
def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(Net())
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')
    parser.add_argument('--label', type=int, default=0)
    # Container environment
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    
    train(parser.parse_args())