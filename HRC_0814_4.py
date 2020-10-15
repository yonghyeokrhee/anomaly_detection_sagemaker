#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(6, 5),
            nn.BatchNorm1d(5),
            nn.LeakyReLU(),
            nn.Linear(5, 4),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
            nn.Linear(4, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
            nn.Linear(4, 5),
            nn.BatchNorm1d(5),
            nn.LeakyReLU(),
            nn.Linear(5, 6),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
    
    
def get_matching_s3_keys(bucket, prefix='', suffix=''):
    """
    Generate the keys in an S3 bucket.

    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch keys that start with this prefix (optional).
    :param suffix: Only fetch keys that end with this suffix (optional).
    """
    s3 = boto3.client('s3')
    kwargs = {'Bucket': bucket}

    # If the prefix is a single string (not a tuple of strings), we can
    # do the filtering directly in the S3 API.
    if isinstance(prefix, str):
        kwargs['Prefix'] = prefix

    while True:

        # The S3 API response is a large blob of metadata.
        # 'Contents' contains information about the listed objects.
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp['Contents']:
            key = obj['Key']
            if key.startswith(prefix) and key.endswith(suffix):
                yield key

        # The S3 API is paginated, returning up to 1000 keys at a time.
        # Pass the continuation token into the next response, until we
        # reach the final page (when this field is missing).
        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break    

def read_data(local_path):
    stats = collections.Counter()   
    with open(local_path, 'r') as f:
        data = []
        timestamp = []
        robot_name = []
        for line in f.readlines():
            stats['total'] += 1
            try:
                data.append(json.loads(line)['data'])
                timestamp.append(json.loads(line)['timestamp'])
                robot_name.append(json.loads(line)['robot_name'])
                stats['success'] += 1
            except:
                stats['err'] += 1
    return pd.DataFrame(data), stats, timestamp, pd.DataFrame(robot_name, columns=['robot_name'])


def _get_train_data_loader(scaler, batch_size, training_dir, is_distributed, **kwargs):
    logger.info("Get train data loader")
    data, stats, timestamp, robot_name = read_data(f'{training_dir}/train.jsonl')
    data = scaler.fit_transform(data).astype(np.float32)
    train_sampler = torch.utils.data.distributed.DistributedSampler(data) if is_distributed else None
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=train_sampler is None,
                                       sampler=train_sampler, **kwargs)


def _get_test_data_loader(scaler, test_batch_size, test_dir, **kwargs):
    # test data dir 명시 필요.
    data, stats, timestamp, robot_name = read_data(f'{test_dir}/train.jsonl')
    data = scaler.transform(data).astype(np.float32)
    logger.info("Get test data loader")
    return torch.utils.data.DataLoader(
        data, batch_size=test_batch_size, shuffle=True, **kwargs
    ), timestamp, robot_name


def _average_gradients(model):
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def train(args):
    scaler = StandardScaler()
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")
    label = args.label

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

    train_loader = _get_train_data_loader(scaler, args.batch_size, args.data_dir, is_distributed, **kwargs)
    test_loader, timestamp, robot_name = _get_test_data_loader(scaler, args.test_batch_size, args.data_dir, **kwargs)

    logger.debug("Processes {}/{} ({:.0f}%) of train data".format(
        len(train_loader.sampler), len(train_loader.dataset),
        100. * len(train_loader.sampler) / len(train_loader.dataset)
    ))

    logger.debug("Processes {}/{} ({:.0f}%) of test data".format(
        len(test_loader.sampler), len(test_loader.dataset),
        100. * len(test_loader.sampler) / len(test_loader.dataset)
    ))

    model = Net().to(device)
    if is_distributed and use_cuda:
        # multi-machine multi-gpu case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = torch.nn.DataParallel(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, data in enumerate(train_loader, 1):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.mse_loss(data , output)
            loss.backward()
            if is_distributed and not use_cuda:
                # average gradients manually for multi-machine cpu case only
                _average_gradients(model)
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.sampler),
                    100. * batch_idx / len(train_loader), loss.item()))
        test(model, test_loader, device, label, timestamp, robot_name, args.model_dir)
    save_model(model, args.model_dir)


def test(model, test_loader, device, label, timestamp, robot_name, model_dir):
    # anomaly score infrence
    model.eval()
    y_pred = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
#             print(output)
            y_pred += [F.mse_loss(data, output, reduction="none").mean(1)]
    y_pred = torch.cat(y_pred).cpu().numpy()
    output = pd.DataFrame(y_pred.astype(str), columns=['Anomaly score'])
    output_example = pd.concat([robot_name, output], axis=1)
    output_example.index = timestamp
    output_example.index.name = 'timestamp'
    print(output_example)
    filename = "output_example.csv"
    output_example.to_csv(filename)
    s3 = boto3.resource('s3')
    s3.meta.client.upload_file(filename, 'hrms-model', 'output_example.csv')
#     s3 = boto3.client('s3')
#     s3.upload_file('output_example', 'hrms-inference', 'outout')
    logger.info(f"Anomaly score {', '.join(y_pred.astype(str))}")


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(Net())
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


def save_model(model, model_dir):
    logger.info("Saving the model.")
    suffix = str(int(pd.Timestamp.now().timestamp()))
    filename = f'model_{suffix}.pth'
    path = os.path.join(model_dir, filename)
    torch.save(model.cpu().state_dict(), path)
    #s3 = boto3.client('s3')
    #s3.upload_file(path, 'hrms-model', f'model_{suffix}.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
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
