import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import os
from shutil import copyfile

import Utils

from preprocess.Dataset import get_dataloader
from MAMBA.model import Transformer
from tqdm import tqdm


def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

    print('[Info] Loading train data...')
    train_data, num_types = load_data(opt.data + 'train.pkl', 'train')
    print('[Info] Loading dev data...')
    dev_data, _ = load_data(opt.data + 'dev.pkl', 'dev')
    print('[Info] Loading test data...')
    test_data, _ = load_data(opt.data + 'test.pkl', 'test')

    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True)
    testloader = get_dataloader(dev_data, opt.batch_size, shuffle=False)
    return trainloader, testloader, num_types


def train_epoch(model, training_data, optimizer, pred_loss_func, opt):
    """ Epoch operation in training phase. """

    model.train()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    _sum_event_ll, _sum_non_event_ll = 0, 0
    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        """ prepare data """
        event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

        """ forward """
        optimizer.zero_grad()
        
        enc_out, prediction = model(event_type, event_time)

        """ backward """
        # negative log-likelihood
        event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
        event_loss = -torch.sum(event_ll - non_event_ll)

        # type prediction
        pred_loss, pred_num_event = Utils.type_loss(prediction[0], event_type, pred_loss_func)

        # time prediction
        se = Utils.time_loss(prediction[1], event_time)

        # SE is usually large, scale it to stabilize training
        scale_time_loss = 100
        loss = event_loss + pred_loss + se / scale_time_loss
        #loss = event_loss
        loss.backward()

        """ update parameters """
        optimizer.step()

        """ note keeping """
        total_event_ll += -event_loss.item()
        total_time_se += se.item()
        total_event_rate += pred_num_event.item()
        total_num_event += event_type.ne(0).sum().item()
        # we do not predict the first event
        total_num_pred += event_type.ne(0).sum().item() - event_time.shape[0]
        _sum_event_ll += event_ll.sum().item()
        _sum_non_event_ll += non_event_ll.sum().item()


    rmse = np.sqrt(total_time_se / total_num_pred)
    print(f"event_ll:{_sum_event_ll / total_num_event: .4f}, non_event_ll: {_sum_non_event_ll / total_num_event: .4f}")
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse


def eval_epoch(model, validation_data, pred_loss_func, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

            """ forward """
            enc_out, prediction = model(event_type, event_time)

            """ compute loss """
            event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
            event_loss = -torch.sum(event_ll - non_event_ll)
            _, pred_num = Utils.type_loss(prediction[0], event_type, pred_loss_func)
            se = Utils.time_loss(prediction[1], event_time)

            """ note keeping """
            total_event_ll += -event_loss.item()
            total_time_se += se.item()
            total_event_rate += pred_num.item()
            total_num_event += event_type.ne(0).sum().item()
            total_num_pred += event_type.ne(0).sum().item() - event_time.shape[0]

    rmse = np.sqrt(total_time_se / total_num_pred)
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse


def train(model, training_data, validation_data, optimizer, scheduler, pred_loss_func, opt):
    """ Start training. """

    valid_event_losses = []  # validation log-likelihood
    valid_pred_losses = []  # validation event type prediction accuracy
    valid_rmse = []  # validation event time prediction RMSE
    best_event_ll = -999999
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        #log_path = opt.log.strip(".txt")
        log_path = opt.log_path
        with open(os.path.join(log_path, opt.log), 'a') as f:
            print('[ Epoch', epoch, ']')
    
            start = time.time()
            train_event, train_type, train_time = train_epoch(model, training_data, optimizer, pred_loss_func, opt)
            f.write('  - (Training)    loglikelihood: {ll: 8.5f}, '
                  'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
                  'elapse: {elapse:3.3f} min\n'
                  .format(ll=train_event, type=train_type, rmse=train_time, elapse=(time.time() - start) / 60))
            print('  - (Training)    loglikelihood: {ll: 8.5f}, '
                  'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
                  'elapse: {elapse:3.3f} min'
                  .format(ll=train_event, type=train_type, rmse=train_time, elapse=(time.time() - start) / 60))
    
            start = time.time()
            valid_event, valid_type, valid_time = eval_epoch(model, validation_data, pred_loss_func, opt)
            f.write('  - (Testing)     loglikelihood: {ll: 8.5f}, '
                  'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
                  'elapse: {elapse:3.3f} min\n'
                  .format(ll=valid_event, type=valid_type, rmse=valid_time, elapse=(time.time() - start) / 60))
            print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
                  'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
                  'elapse: {elapse:3.3f} min'
                  .format(ll=valid_event, type=valid_type, rmse=valid_time, elapse=(time.time() - start) / 60))
            if valid_event > best_event_ll:
               best_event_ll = valid_event
               #torch.save(model.state_dict(), os.path.join(log_path, "model.pt"))
               #f.write("---(Best model save)----\n")
               print("---(Best model save)----")
    
            valid_event_losses += [valid_event]
            valid_pred_losses += [valid_type]
            valid_rmse += [valid_time]
            f.write('  - [Info] [Valid] Maximum ll: {event: 8.5f}, '
                  'Maximum accuracy: {pred: 8.5f}, Minimum RMSE: {rmse: 8.5f}\n'
                  .format(event=max(valid_event_losses), pred=max(valid_pred_losses), rmse=min(valid_rmse)))
            print('  - [Info] [Valid] Maximum ll: {event: 8.5f}, '
                  'Maximum accuracy: {pred: 8.5f}, Minimum RMSE: {rmse: 8.5f}'
                  .format(event=max(valid_event_losses), pred=max(valid_pred_losses), rmse=min(valid_rmse)))
    
            # logging
            f.write('-[Valid][Current] {epoch}, {ll: 8.5f}, {acc: 8.5f}, {rmse: 8.5f}\n'
                    .format(epoch=epoch, ll=valid_event, acc=valid_type, rmse=valid_time))
            print('-[Valid][Current] {epoch}, {ll: 8.5f}, {acc: 8.5f}, {rmse: 8.5f}'
                    .format(epoch=epoch, ll=valid_event, acc=valid_type, rmse=valid_time))

        scheduler.step()


def main():
    """ Main function. """

    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)

    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=16)

    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_rnn', type=int, default=256)
    parser.add_argument('-d_inner_hid', type=int, default=128)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)

    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-smooth', type=float, default=0.1)

    parser.add_argument('-log', type=str, default='log.txt')
    
    parser.add_argument('-seed', type=int, default=1)
    opt = parser.parse_args()

    # default device is CUDA
    opt.device = torch.device('cuda')

    # fix random seed
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # setup the log file
    #log_path = opt.log.strip(".txt")
    log_path = os.path.join(opt.data, "Logs", "gatech")
    opt.log_path = log_path
    os.makedirs(log_path, exist_ok=True)
    abspath = os.path.abspath(log_path)
    copyfile("run.sh", os.path.join(log_path, "run.sh"))
    with open(os.path.join(log_path, "log.txt"), "w") as f_spec:
        f_spec.write("Training Log\nSpecifications\n")
        for k, v in vars(opt).items():
            f_spec.write(f"{k} : {v}\n")
        f_spec.write(f"PathDomain : {abspath}\nPathLog : {os.path.join(abspath, 'log.txt')}\nPathSave : {os.path.join(abspath, 'model.pt')}\n")
    with open(os.path.join(log_path, opt.log), 'w') as f:
        f.write('Epoch, Log-likelihood, Accuracy, RMSE\n')

        f.write('[Info] parameters: {}\n'.format(opt))

        """ prepare dataloader """
        trainloader, testloader, num_types = prepare_dataloader(opt)
    
        """ prepare model """
        model = Transformer(
            num_types=num_types,
            d_model=opt.d_model,
            d_rnn=opt.d_rnn,
            d_inner=opt.d_inner_hid,
            n_layers=opt.n_layers,
            n_head=opt.n_head,
            d_k=opt.d_k,
            d_v=opt.d_v,
            dropout=opt.dropout,
        )
        model.to(opt.device)
    
        """ optimizer and scheduler """
        optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                               opt.lr, betas=(0.9, 0.999), eps=1e-05)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)
    
        """ prediction loss function, either cross entropy or label smoothing """
        if opt.smooth > 0:
            pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, num_types, ignore_index=-1)
        else:
            pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
    
        """ number of parameters """
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('[Info] Number of parameters: {}'.format(num_params))
        f.write('[Info] Number of parameters: {}\n'.format(num_params))
    
        """ train the model """
    train(model, trainloader, testloader, optimizer, scheduler, pred_loss_func, opt)


if __name__ == '__main__':
    main()
