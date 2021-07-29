
from Model import TransImgPred
import itertools
from torchvision import models
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from dataset import lonData
import torch.optim as optim
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score, f1_score
import numpy as np
import time
import shutil
import os
import argparse
from focal_loss import FocalLoss
import utils
import pdb
import Model
import torchvision

device = torch.device('cuda:0')

## ===================================== argparse ==============================================

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='test')
parser.add_argument('--checkpoints_dir', default='Model_and_Result')
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

# parser.add_argument('--cls_num', default=3, type=int)
parser.add_argument('--label_type', default='pre_diabet', type=str)
parser.add_argument('--nworkers', default=8, type=int)
parser.add_argument('--feat_net', default='self', type=str)

# model
parser.add_argument('--feature_merge', action='store_true', help='if specified, use feature merge option of transformer decoder')
parser.add_argument('--single_pred', action='store_true', help='if specified, use single output of transformer decoder to evaluate')
parser.add_argument('--focal', action='store_true', help='if specified, use single output of transformer decoder to evaluate')


# training parameters
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs with the initial learning rate')
parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
parser.add_argument('--clip_grad', action='store_true', help='if specified, parameter grad clipped')
parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument('--lr', default=0.0002, type=float)
parser.add_argument('--lr_step', type=int, default=50, help='multiply by a gamma every lr_step iterations')
parser.add_argument('--lr_ratio', type=int, default=0.95, help='gamma multiplied every lr_step iterations')
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--batch_size', default=32, type=int)

parser.add_argument('--criteria', default='val_loss')
parser.add_argument('--stop_tolerance', default=200, type=int)

args = parser.parse_args()


#utils.print_options(parser, args)


## ===================================== file saving settings ==================================

load_model = None
save_model = True

exp_name = args.name
result_dir = args.checkpoints_dir + '/' + exp_name
model_directory = result_dir + '/models'

if os.path.exists(result_dir):
    shutil.rmtree(result_dir)

if not os.path.exists(model_directory):
    os.makedirs(model_directory)

utils.print_options(parser, args)

shutil.copy('focal_loss.py', result_dir)
shutil.copy('utils.py', result_dir)
shutil.copy('dataset.py', result_dir)
shutil.copy('Model.py', result_dir)
shutil.copy(__file__, result_dir)

## ===================================== loss parameters =======================================

# fl_alpha = torch.tensor([1.0,1.0,1.0]).to(device)
fl_alpha = None
fl_gamma = 2

## ===================================== training parameters =================================== ##

feature_size = 256
img_len = 2
nhead = 2
nlayers = 2

lr = args.lr
epochs = args.n_epochs + args.n_epochs_decay
dropout = args.dropout
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

trn_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
                             batch_size=args.batch_size, shuffle=True)

tst_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=args.batch_size, shuffle=True)

## ==================================== feature model & longitudinal model ===========================
flag_net = args.feat_net
if flag_net == 'self':
    net_fea = Model.featureNet()
    net_fea.to(device)
    num_fits = 2048
    net = Model.downFeaNet_mnist(num_fits=2048, feature_size=feature_size)
    pre_epoch = 10
elif flag_net == 'resnet':
    net = models.resnet50(pretrained=True)
    num_fits = net.fc.in_features
    # net.fc = nn.Linear(num_fits, feature_size)
    feature_size = num_fits
    net.fc = nn.Flatten()
elif flag_net == 'simple':
    net = Model.simpleFeatNet(feature_size=feature_size)




net = net.to(device)
#
# input_data = torch.zeros((1, 3, 224,224))
# net(input_data)


## =================================== criterion & setting ===========================================

if not args.focal:
    criterion = nn.CrossEntropyLoss()
    criterion_val = nn.CrossEntropyLoss()
else:
    criterion = FocalLoss(alpha=fl_alpha, gamma=fl_gamma)
    # criterion_val = nn.CrossEntropyLoss()
    criterion_val = nn.CrossEntropyLoss()

# optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer = optim.Adam(itertools.chain(net.parameters(), net_fea.parameters()), lr=lr, betas=(0.5, 0.999))
scheduler = utils.get_scheduler(optimizer, args)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_step, gamma=lr_ratio)

## ======================================= save or load model ===========================================



## ======================================= start training ===============================================

with open(result_dir + '/training_log.txt', 'w') as f:
    f.close()

best_loss = np.inf
stop_step = 0
best_acc = 0

# pdb.set_trace()

print('Training START !!!')

for epoch in range(epochs):

    lossAll, n_exp = 0, 0
    labelAll, predAll = [], []
    # i=0

    net.train()
    start = time.time()
    if epoch < pre_epoch :
        net_fea.eval()
    else:
        net_fea.train()

    # data = utils.load_var('1.pkl')
    # for i in range(200):
    for i, data in enumerate(trn_loader):
        
        end1 = time.time()
        # print(end1-start)

        img, label = data
        img, label = img.to(device), label.to(device)
        img = torch.cat((img,img,img),dim=1)
        lossBlock, b_exp = 0, 0

        if args.feat_net == 'self':
            if epoch < pre_epoch :
                with torch.no_grad():
                    img_fea = net_fea(img)
            else:
                img_fea = net_fea(img)
            img_fea, img_cls = net(img_fea)

        # end2 = time.time()

        # change to shape [seq_len, bz, fea_len]


        optimizer.zero_grad()

        loss = criterion(img_cls, label)

        pred = img_cls.argmax(1).cpu().numpy()
        predAll += pred.tolist()
        labelAll += (label.cpu().numpy()).tolist()

        loss.backward()
        optimizer.step()

        # end3 = time.time()

        lossAll += loss.item() * len(label)
        lossBlock += loss.item() * len(label)
        n_exp += len(label)
        b_exp += len(label)

        # print(end2-end1, end3-end2, end1-start)

        if i % int(len(trn_loader) / 5) == 0 and i >0:
            end_time = time.time()
            cur_loss = lossBlock / b_exp
            print('| epoch {:3d} | {:5d}/{:5d} batches, time: {:5.4f} | lr {:02.6f} | loss {:5.4f} | cls_ratio {:4d} / {:4d} / {:4d}'.format(
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

    net.eval()
    net_fea.eval()

    # data = utils.load_var('2.pkl')
    # for i in range(1):
    for i, data in enumerate(tst_loader):

        total_loss = 0

        img_val, label_val = data
        img_val, label_val =img_val.to(device), label_val.to(device)
        img_val = torch.cat((img_val, img_val, img_val),dim=1)

        with torch.no_grad():
            
            if args.feat_net == 'self':

                img_fea_val = net_fea(img_val)
                img_fea_val, img_val_cls = net(img_fea_val)
            else:
                img_fea_val, img_val_cls = net(img_val)


            pred = img_val_cls.argmax(1).cpu().numpy()
            predAll += pred.tolist()
            labelAll += (label_val.cpu().numpy()).tolist()

            loss_val = criterion_val(img_val_cls, label_val)

            lossAll += loss_val.item() * len(label_val)
            n_exp += len(label_val)


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

