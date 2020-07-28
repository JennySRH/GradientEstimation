import os

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms


class Binarize(object):
    def __init__(self):
        pass

    def __call__(self, x):
        u = torch.rand(x.shape)
        x = (x > u).type(torch.float32)
        return x

class Flatten(object):
    def __init__(self):
        pass

    def __call__(self, x):
        return torch.reshape(x, (x.shape[0], -1)).squeeze()

# lots of codes adapted from
# https://github.com/jmtomczak/vae_vampprior/blob/master/utils/load_data.py
def load_static_mnist(bs, test_bs, **kwargs):
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])

    with open(os.path.join('datasets', 'staticMNIST', 'binarized_mnist_train.amat')) as f:
        lines = f.readlines()
    x_train = torch.from_numpy(lines_to_np_array(lines).astype(np.float)).type(torch.float)
    with open(os.path.join('datasets', 'staticMNIST', 'binarized_mnist_valid.amat')) as f:
        lines = f.readlines()
    x_val = torch.from_numpy(lines_to_np_array(lines).astype(np.float)).type(torch.float)
    with open(os.path.join('datasets', 'staticMNIST', 'binarized_mnist_test.amat')) as f:
        lines = f.readlines()
    x_test = torch.from_numpy(lines_to_np_array(lines).astype(np.float)).type(torch.float)

    mean_obs = torch.mean(x_train, dim=0)
    y_train = torch.from_numpy(np.zeros((x_train.shape[0], 1))).type(torch.float)
    y_val = torch.from_numpy(np.zeros((x_val.shape[0], 1))).type(torch.float)
    y_test = torch.from_numpy(np.zeros((x_test.shape[0], 1))).type(torch.float)

    train_set = torch.utils.data.TensorDataset(x_train, y_train)
    val_set = torch.utils.data.TensorDataset(x_val, y_val)
    test_set = torch.utils.data.TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=test_bs, shuffle=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=test_bs, shuffle=True, **kwargs)

    return train_loader, val_loader, test_loader, mean_obs


def load_dynamic_mnist(bs, test_bs, **kwargs):
    train_set = datasets.MNIST('datasets/dynamic_mnist', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        Binarize(),
        Flatten()
    ]))
    train_sampler = SubsetRandomSampler(list(range(0, 50000)))
    val_sampler = SubsetRandomSampler(list(range(50000, 60000)))
    test_set = datasets.MNIST('datasets/dynamic_mnist', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        Binarize(),
        Flatten()
    ]))
    train_loader = DataLoader(train_set, batch_size=bs, sampler=train_sampler, **kwargs)
    mean_obs = 0.0
    for img, _ in train_loader:
        mean_obs += img.sum(0)
    mean_obs /= len(train_loader.dataset)
    val_loader = DataLoader(train_set, batch_size=test_bs, sampler=val_sampler, **kwargs)
    test_loader = DataLoader(test_set, batch_size=test_bs, shuffle=True, **kwargs)

    return train_loader, val_loader, test_loader, mean_obs


def load_omniglot(bs, test_bs, **kwargs):
    omniglot_raw = loadmat(os.path.join('datasets', 'omniglot', 'chardata.mat'))
    train_data = omniglot_raw['data'].T.astype('float32').reshape((-1, 28, 28)).reshape((-1, 784), order='F')
    test_data = omniglot_raw['testdata'].T.astype('float32').reshape((-1, 28, 28)).reshape((-1, 784), order='F')
    val_data = train_data[-1345:]
    train_data = train_data[:-1345]
    y_train = np.zeros((train_data.shape[0], 1))
    y_val = np.zeros((val_data.shape[0], 1))
    y_test = np.zeros((test_data.shape[0], 1))
    train_set = torch.utils.data.TensorDataset(Binarize()(torch.from_numpy(train_data)), torch.from_numpy(y_train))
    test_set = torch.utils.data.TensorDataset(Binarize()(torch.from_numpy(test_data)), torch.from_numpy(y_test))
    val_set = torch.utils.data.TensorDataset(Binarize()(torch.from_numpy(val_data)), torch.from_numpy(y_val))
    mean_obs = torch.tensor(np.mean(train_data,axis=0))
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=test_bs, shuffle=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=test_bs, shuffle=True, **kwargs)

    return train_loader, val_loader, test_loader, mean_obs


def load_caltech101silhouettes(bs, test_bs, **kwargs):
    def reshape_data(data):
        return data.reshape((-1, 28, 28)).reshape((-1, 28 * 28), order='F')

    caltech_raw = loadmat(os.path.join('datasets', 'Caltech101', 'caltech101_silhouettes_28_split1.mat'))
    x_train = 1. - reshape_data(caltech_raw['train_data'].astype('float32'))
    x_val = 1. - reshape_data(caltech_raw['val_data'].astype('float32'))
    x_test = 1. - reshape_data(caltech_raw['test_data'].astype('float32'))
    mean_obs = torch.tensor(np.mean(x_train, axis=0))
    y_train = caltech_raw['train_labels']
    y_val = caltech_raw['val_labels']
    y_test = caltech_raw['test_labels']

    train = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=True, **kwargs)

    validation = torch.utils.data.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = DataLoader(validation, batch_size=test_bs, shuffle=False, **kwargs)

    test = torch.utils.data.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = DataLoader(test, batch_size=test_bs, shuffle=True, **kwargs)

    return train_loader, val_loader, test_loader, mean_obs


def load_dataset(args):
    if args.dataset == 'dynamic_mnist':
        return load_dynamic_mnist(args.batch_size, args.test_batch_size)
    elif args.dataset == 'static_mnist':
        return load_static_mnist(args.batch_size, args.test_batch_size)
    elif args.dataset == 'omniglot':
        return load_omniglot(args.batch_size, args.test_batch_size)
    elif args.dataset == 'caltech101':
        return load_caltech101silhouettes(args.batch_size, args.test_batch_size)
    else:
        raise Exception('Invalid Dataset Name!')
