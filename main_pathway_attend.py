"""
Attention mechanism for personalised genomics

Adham Beyki
Deakin University
PRaDA, A2I2 - Deakin Univesity
2018-11-19
"""

import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import scale
from sklearn import metrics
from sklearn.model_selection import train_test_split

from logger import Logger


SEED = 100
np.random.seed(SEED)
torch.manual_seed(SEED)
USE_GPU = torch.cuda.is_available()
DEVICE = 0

def to_var(x):
    if USE_GPU:
        x = x.cuda(DEVICE)
    return Variable(x)

def to_np(x):
    return x.data.cpu().numpy()

def to_class(x):
    return to_np(x.topk(1)[1]).ravel()


def load_data(data):

    c2i = {'LumA': 1, 'LumB': 0}

    data = pd.read_pickle(data)
    pathway_df = data['pathway_df']
    pathway_df['PAM50'] = pathway_df['PAM50'].apply(lambda x: c2i[x])
    train_idxs = data['train_idxs']
    test_idxs = data['test_idxs']

    cols = pathway_df.columns[:-2]

    X_train = pathway_df.loc[train_idxs][cols].values
    y_train = pathway_df.loc[train_idxs]['PAM50'].values
    X_test = pathway_df.loc[test_idxs][cols].values
    y_test = pathway_df.loc[test_idxs]['PAM50'].values

    # normalize per person
    means = X_train.mean(1)
    X_train = X_train/means[:, None]
    means = X_test.mean(1)
    X_test = X_test/means[:, None]

    # scale
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    X_train = (X_train-means) / stds
    X_test = (X_test-means) / stds

    return (X_train, y_train), (X_test, y_test)


def get_dataloader(X, y, batch_size, shuffle=True):
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    ds = TensorDataset(X_tensor, y_tensor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return dl

def print_results(y_true, y_pred):
    print(metrics.classification_report(y_true, y_pred))
    print(metrics.confusion_matrix(y_true, y_pred))


class Net(nn.Module):
    def __init__(self, n_input, n_class, p_dropout=0.5):
        super(Net, self).__init__()

        N_PRJ = 64
        N_FEATS = 64

        M = np.random.rand(n_input, N_PRJ)
        M = scale(M)
        self.M = torch.from_numpy(M).type(torch.FloatTensor)
        if USE_GPU:
            self.M = self.M.cuda(DEVICE)

        self.f_x = nn.Sequential(
            nn.Linear(N_PRJ, N_FEATS),
            nn.Tanh(),
            nn.Dropout(p_dropout)
        )

        self.f_attn = nn.Sequential(
            nn.Linear(N_PRJ, 32),
            nn.Tanh(),
            nn.Dropout(p_dropout),
            nn.Linear(32, 1),
            nn.Softmax(dim=1)
        )

        self.f_y = nn.Sequential(
            nn.BatchNorm1d(N_FEATS),
            nn.Linear(N_FEATS, N_FEATS//2),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(N_FEATS//2, n_class),
            nn.LogSoftmax(dim=1)
        )


    def forward(self, x):
        x1 = x[:, :, None] * self.M

        unattn_feats = self.f_x(x1)
        betas = self.f_attn(x1)
        # check sum() instead of mean()
        feats = (betas*unattn_feats).mean(1)

        out = self.f_y(feats)
        return out, betas


def train(epoch, train_loader, model, criterion, optimizer):
    """train for one epoch
    """

    #switch model to training mode
    model.train()

    n_samples = 0
    epoch_loss = 0
    epoch_outputs = []
    epoch_targets = []

    for iteration, batch in enumerate(train_loader, 1):
        batch_X, batch_targets = batch
        batch_size = batch_X.size(0)
        epoch_targets.append(to_np(batch_targets))

        n_samples += batch_size

        # train on batch data
        batch_loss, batch_outputs = train_batch(batch, model, criterion, optimizer)
        epoch_outputs.append(batch_outputs)
        epoch_loss += batch_loss.item()

    # compute and report average loss
    avg_loss = epoch_loss / n_samples
    f1_score = metrics.f1_score(
        np.hstack(epoch_targets),
        np.hstack(epoch_outputs),
        average='weighted'
    )
    print('\r===> Epoch {} Train, Avg. Loss: {:.4f}, f1_score: {:.4f}'.format(epoch, avg_loss, f1_score))
    print_results(
        np.hstack(epoch_targets),
        np.hstack(epoch_outputs)
    )

    return avg_loss, f1_score


def train_batch(batch, model, criterion, optimizer):
    """train for one batch
    """

    batch_X, batch_targets = batch

    # clear gradient accumu;ators
    optimizer.zero_grad()

    # forward pass
    batch_outputs, _ = model(to_var(batch_X))

    # loss
    loss = criterion(batch_outputs, to_var(batch_targets))

    # backprop and update params
    loss.backward()
    optimizer.step()

    return loss, to_class(batch_outputs)


def test(epoch, test_loader, model, criterion):
    """evaluate the model with test dataset
    """
    # switch to evaluation mode
    model.eval()

    n_samples = 0
    epoch_loss = 0
    epoch_outputs = []
    epoch_targets = []

    # loop through batches in test_loader
    for iteration, batch in enumerate(test_loader, 1):

        batch_X, batch_targets = batch
        batch_size = batch_X.size(0)
        epoch_targets.append(to_np(batch_targets))

        n_samples += batch_size

        # forward
        batch_outputs, _ = model(to_var(batch_X))
        epoch_outputs.append(to_class(batch_outputs))

        # loss
        batch_loss = criterion(batch_outputs, to_var(batch_targets))
        epoch_loss += batch_loss.item()

    # compute and report average loss
    avg_loss = epoch_loss / n_samples
    f1_score = metrics.f1_score(
        np.hstack(epoch_targets),
        np.hstack(epoch_outputs),
        average='weighted'
    )
    print('\r===> Epoch {} Test,  Avg. Loss: {:.4f}, f1_score: {:.4f}'.format(epoch, avg_loss, f1_score))
    print_results(
        np.hstack(epoch_targets),
        np.hstack(epoch_outputs)
    )

    return avg_loss, f1_score


def get_model_output(data_loader, model):
    # switch to evaluation mode
    model.eval()

    model_preds = []
    betas = []

    # loop through batches in data_loader
    for iteration, batch in enumerate(data_loader, 1):

        batch_X, batch_targets = batch
        batch_outputs, batch_betas = model(to_var(batch_X))

        model_preds.append(to_class(batch_outputs))
        betas.append(to_np(batch_betas))

        print('\r===> preparing model outputs ({}/{})'.format(iteration, len(data_loader)), end='')

    model_preds = np.hstack(model_preds)
    betas = np.concatenate(betas)

    return model_preds, betas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='path to input data')
    parser.add_argument('--bsz', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--tb', default=None, help='tensorboard log name')
    parser.add_argument('--output', default=None, help='output file')
    args = parser.parse_args()
    print(args); print()

    # load data
    print('==> loading data'); print()
    (X_train, y_train), (X_test, y_test) = load_data(args.data)
    train_loader = get_dataloader(X_train, y_train, args.bsz)
    test_loader = get_dataloader(X_test, y_test, args.bsz)

    # model
    print()
    print('==> building model'); print()
    n_input = X_train.shape[1]
    n_class = np.unique(y_train).shape[0]
    model = Net(n_input, n_class)
    class_weights = [0.4, 0.6]
    class_weights = torch.FloatTensor(class_weights).cuda() if USE_GPU else torch.FloatTensor(class_weights)
    criterion = nn.NLLLoss(size_average=False, weight=class_weights)
    if USE_GPU:
        model = model.cuda()
        criterion = criterion.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(model)

    # instantiate tensorboard logger
    if args.tb is not None:
        logger_name = './.tb_logs/{}'.format(
            '_'.join([__file__.split('.')[0], args.tb])
        )
        logger = Logger(logger_name)
        print('tensorboard logger name: {}'.format(logger_name))

    # train the network
    for epoch in range(1, args.epochs+1):
        info = {}

        info['train_loss'], info['train_f1'] = train(epoch, train_loader, model, criterion, optimizer)
        info['test_loss'], info['test_f1'] = test(epoch, test_loader, model, criterion)
        print()

        # log to tensorboard
        if args.tb is not None:
            for k, v in info.items():
                logger.scalar_summary(k, v, epoch)


    # save model outputs
    if args.output is not None:
        data_loader = get_dataloader(X_test, y_test, args.bsz, shuffle=False)
        preds, betas = get_model_output(data_loader, model)
        pd.to_pickle(
            {
                'preds': preds,
                'betas': betas
            }, args.output
        )


if __name__ == '__main__':
    main()
