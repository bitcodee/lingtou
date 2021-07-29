import model
from model import TransImgPred
import itertools
from torchvision import models
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from dataset import lonData, lonData_resnet
import torch.optim as optim
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score, f1_score
import numpy as np
import time
import shutil
import os
import argparse
from focal_loss import FocalLoss
import utils

device = torch.device('cuda:0')

## ===================================== argparse ==============================================

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='test_pc')

# parser.add_argument('--cls_num', default=3, type=int)
parser.add_argument('--label_type', default='diag', type=str)
parser.add_argument('--nworkers', default=8, type=int)

# model
parser.add_argument('--feature_merge', action='store_true', help='if specified, use feature merge option of transformer decoder')
parser.add_argument('--single_pred', action='store_true', help='if specified, use single output of transformer decoder to evaluate')

# training parameters
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--lr_step', type=int, default=50, help='multiply by a gamma every lr_step iterations')
parser.add_argument('--lr_ratio', type=int, default=0.95, help='gamma multiplied every lr_step iterations')
parser.add_argument('--batch_size', default=16, type=int)

parser.add_argument('--criteria', default='val_loss')
parser.add_argument('--stop_tolerance', default=200, type=int)

args = parser.parse_args()

print(args)

## ===================================== file saving settings ==================================

load_model = None
save_model = True

exp_name = args.exp_name
result_dir = 'Model_and_Result' + '/resnet/' + exp_name
model_directory = result_dir + '/models'

if os.path.exists(result_dir):
    shutil.rmtree(result_dir)

if not os.path.exists(model_directory):
    os.makedirs(model_directory)

shutil.copy('focal_loss.py', result_dir)
shutil.copy('utils.py', result_dir)
shutil.copy('dataset.py', result_dir)
shutil.copy('model.py', result_dir)
shutil.copy(__file__, result_dir)

## ===================================== loss parameters =======================================

fl_alpha = torch.Tensor([1.0,1.0]).to(device)
fl_gamma = 0

## ===================================== training parameters =================================== ##

feature_size = 256
img_len = 2

lr = args.lr
epochs = args.n_epochs + args.n_epochs_decay
dropout = 0.2
batch_size = args.batch_size
batch_size_val = 128

nworkers = args.nworkers

stop_criteria = args.criteria
early_stop_tolerance = args.stop_tolerance

## ==========================================  data =================================================

if args.label_type == 'pre_diabet':
    cls_num = 3
elif args.label_type == 'pre_diabet_2':
    cls_num = 2
elif args.label_type == 'diag':
    cls_num = 2    

trn_data = lonData_resnet('./data', 'train', img_size=224, label_type=args.label_type)
trn_loader = DataLoader(trn_data, batch_size=batch_size, shuffle=True , num_workers=nworkers)
tst_data = lonData_resnet('./data', 'test', img_size=224, label_type=args.label_type)
tst_loader = DataLoader(tst_data, batch_size=batch_size, shuffle=False, num_workers=nworkers)

## ==================================== feature model & longitudinal model ===========================

# net = models.resnet50(pretrained=True)
# num_fits = net.fc.in_features
# net.fc = nn.Linear(num_fits, feature_size)

net = model.featureNet()
net = net.to(device)

#
# input_data = torch.zeros((1, 3, 224,224))
# net(input_data)


## =================================== criterion & setting ===========================================

criterion = nn.CrossEntropyLoss()
# criterion = FocalLoss(alpha=fl_alpha, gamma=fl_gamma)

optimizer = optim.Adam(net.parameters(), lr=lr)
# optimizer = optim.Adam(itertools.chain(net.parameters(),model.parameters()), lr=lr)
scheduler = utils.get_scheduler(optimizer, args)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_step, gamma=lr_ratio)

## ======================================= save or load model ===========================================


## ======================================= start training ===============================================

with open(result_dir + '/training_log.txt', 'w') as f:
    f.close()

best_loss = np.inf
stop_step = 0
best_acc = 0

print('Training START !!!')

for epoch in range(epochs):

    lossAll, n_exp = 0, 0
    labelAll, predAll = [], []
    # i=0

    start = time.time()
    for i, data in enumerate(trn_loader):
        # for data in trn_loader:
        end1 = time.time()
        # print(end1-start)
        net.train()
        lossBlock, b_exp = 0, 0

        # img_path[0][*] and img_path[1][*] are a pair, e.g. 2012_R2.jpg, 2013_R2.jpg
        base_seq, img_seq, label_seq, img_path, label_path = data
        base_seq, img_seq, label_seq = base_seq.to(device), img_seq.to(device), label_seq.to(device)
        # merge batch and sequence together [bz, seq_len] -> [bz*seq_len]
        label_seq_vec = label_seq.reshape((label_seq.size(0)*img_len))


        # label = {0, 1, 2}; 0- normal, 1- slightly, 2- seriously

        base_size, img_size = base_seq.size(), img_seq.size()

        
        base_seq = torch.reshape(base_seq, (-1, *(base_size[2:])))
        img_seq = torch.reshape(img_seq, (-1, *(img_size[2:])))

        train_seq = img_seq
        
        # import pdb;pdb.set_trace()
        train_fea, train_cls = net(train_seq)

        optimizer.zero_grad()

        loss = criterion(train_cls, label_seq_vec)

        pred = train_cls.argmax(1).cpu().numpy()
        predAll += pred.tolist()
        labelAll += (label_seq_vec.cpu().numpy()).tolist()

        loss.backward()
        optimizer.step()

        # end3 = time.time()

        lossAll += loss.item() * len(label_seq_vec)
        lossBlock += loss.item() * len(label_seq_vec)
        n_exp += len(label_seq_vec)
        b_exp += len(label_seq_vec)

        # print(end2-end1, end3-end2, end1-start)

        if i % int(len(trn_loader) / 5) == 0 and i >0:
            end_time = time.time()
            cur_loss = lossBlock / b_exp
            print('| epoch {:3d} | {:5d}/{:5d} batches, time: {:5.4f} | lr {:02.4f} | loss {:5.4f} | cls_ratio {:4d} / {:4d} / {:4d}'.format(
                epoch+1, i, len(trn_loader), end_time-start, scheduler.get_last_lr()[0], cur_loss, labelAll.count(0), labelAll.count(1), labelAll.count(2)))
            interval_loss = 0
            # with open(result_dir + '/training_log.txt', 'a') as f:
            #     f.write('| epoch {:3d} | {:5d}/{:5d} batches, time: {:5.4f} | lr {:02.6f} | loss {:5.6f} | cls_ratio {:4d} / {:4d} / {:4d}'.format(
            #         epoch+1, i, len(trn_loader), end_time-start, scheduler.get_last_lr()[0], cur_loss, labelAll.count(0), labelAll.count(1), labelAll.count(2)))

        # i += 1


    scheduler.step()

    train_loss = lossAll / n_exp
    lossAll, n_exp = 0, 0

    predAll = []
    labelAll = []

    for i, data in enumerate(tst_loader):
        net.eval()
        total_loss = 0

        base_val, img_val, label_val, img_path, label_path = data
        base_val, img_val, label_val = base_val.to(device), img_val.to(device), label_val.to(device)
        label_val_vec = label_val.reshape((label_val.size(0)*img_len))

        base_size, img_size = base_val.size(), img_val.size()

        base_val = torch.reshape(base_val, (-1, *(base_size[2:])))
        img_val = torch.reshape(img_val, (-1, *(img_size[2:])))

        val_seq = img_val

        with torch.no_grad():
            val_fea, val_cls = net(val_seq)

            


            pred = val_cls.argmax(1).cpu().numpy()
            predAll += pred.tolist()       
            labelAll += (label_val_vec.cpu().numpy()).tolist()

            loss_val = criterion(val_cls, label_val_vec)

            lossAll += loss_val.item() * len(label_val_vec)
            n_exp += len(label_val_vec)


    # pred = np.concatenate(predAll, axis=0)

    acc = accuracy_score(labelAll, predAll)
    kappa = cohen_kappa_score(labelAll, predAll)
    f1 = f1_score(labelAll, predAll, average='macro')
    val_loss = lossAll / n_exp

    end_time = time.time()
    print('| epoch {:3d} | time {:5.4f} | train_loss {:5.4f}, val_loss {:5.4f} | acc {:5.4f}, kappa {:5.4f}, F1 {:5.4f} | cls_ratio {:4d} / {:4d} / {:4d}'.format(
        epoch+1, end_time-start, train_loss, val_loss, acc, kappa, f1, labelAll.count(0), labelAll.count(1), labelAll.count(2)))
    with open(result_dir + '/training_log.txt', 'a') as f:
        f.write('| epoch {:3d} | time {:5.4f} | train_loss {:5.4f}, val_loss {:5.4f}| acc {:5.4f}, kappa {:5.4f}, F1 {:5.4f} | cls_ratio {:4d} / {:4d} / {:4d} \n'.format(
            epoch+1, end_time-start, train_loss, val_loss, acc, kappa, f1, labelAll.count(0), labelAll.count(1), labelAll.count(2)))

    if (epoch+1) % 20 == 0:
        torch.save(net.state_dict(), model_directory + '/feaNet_' + str(epoch+1) + '.pth')


    if stop_criteria == 'val_loss':
        if val_loss < best_loss-0.0002:
            best_loss = val_loss
            best_acc = acc
            best_kappa = kappa
            stop_step = 0
            best_epoch = epoch

            torch.save(net.state_dict(), model_directory + '/feaNet_best.pth')

        else:
            stop_step += 1
            if stop_step > early_stop_tolerance:
                # print('Early stopping is trigger at epoch: %2d. ----->>>>> Best loss: %.4f, acc: %.4f at epoch %2d (step = %2d)'
                #       %(epoch+1, best_loss, best_acc, best_epoch+1, best_global_step))
                #
                # with open(result_dir + '/training_log.txt', 'a') as text_file:
                #     text_file.write(
                #         'Early stopping is trigger. ----->>>>> Best loss: %.4f, acc: %.4f at epoch %2d (step = %2d)\n'
                #         % (best_loss, best_acc, best_epoch+1, best_global_step))
                # s = open(model_directory + '/checkpoint').read()
                # s = s.replace('model_checkpoint_path: "model.ckpt-' + str(training_util.global_step(sess, global_step)) + '"', 'model_checkpoint_path: "model.ckpt-' + str(best_global_step) +'"')
                # f = open(model_directory + '/checkpoint', 'w')
                # f.write(s)
                # f.close()
                break
    elif stop_criteria == 'val_acc':
        if (best_acc < acc) or (abs(best_acc - acc) < 0.0001 and val_loss < best_loss):
            best_acc = acc
            best_loss = val_loss
            best_kappa = kappa
            best_epoch = epoch

            torch.save(net.state_dict(), model_directory + '/feaNet_best.pth')



print('Early stopping is trigger. ----->>>>>> Best loss: {:.4f}, acc: {:.4f}, kappa: {:.4f} at epoch {:3d}'.format(
    best_loss, best_acc, best_kappa, best_epoch+1))
with open(result_dir + '/training_log.txt', 'a') as f:
    f.write('Early stopping is trigger. ----->>>>>> Best loss: {:.4f}, acc: {:.4f}, kappa: {:.4f} at epoch {:3d}'.format(
        best_loss, best_acc, best_kappa, best_epoch+1))


    # # load
    # model.load_state_dict(torch.load(model_directory + '/transNet.pth'))
    # net.load_state_dict(torch.load(model_directory + '/feaNet.pth'))