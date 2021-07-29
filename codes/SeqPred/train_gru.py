
from Model import TransImgPredEn
import itertools
from torchvision import models
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import dataset
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
import random

device = torch.device('cuda:0')

## ===================================== argparse ==============================================

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='test')
parser.add_argument('--checkpoints_dir', default='Model_and_Result')
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

# parser.add_argument('--cls_num', default=3, type=int)
parser.add_argument('--label_type', default='pre_diabet', type=str)
parser.add_argument('--nworkers', default=4, type=int)
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
parser.add_argument('--lr', default=0.00002, type=float)
parser.add_argument('--lr_step', type=int, default=50, help='multiply by a gamma every lr_step iterations')
parser.add_argument('--lr_ratio', type=int, default=0.95, help='gamma multiplied every lr_step iterations')
parser.add_argument('--dropout', default=0, type=float)
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

feature_size = 64
feat_len = feature_size #* 7 * 7
img_len = 6
nhead = 4
nlayers = 1
hid = 64

lr = args.lr
epochs = args.n_epochs + args.n_epochs_decay
dropout = args.dropout
batch_size = args.batch_size
batch_size_val = 8

nworkers = args.nworkers

stop_criteria = args.criteria
early_stop_tolerance = args.stop_tolerance

## ==========================================  data =================================================

if args.label_type == 'pre_diabet':
    cls_num = 3
elif args.label_type == 'pre_diabet_2':
    cls_num = 2
elif args.label_type == 'diag':
    cls_num = 1
elif args.label_type == 'glucose':
    cls_num = 3

seed = random.randint(1, 5000)
print(seed)
with open(result_dir + '/train_opt.txt', 'a') as f:
    f.close()


trn_list, tst_list = utils.idList(label_type=args.label_type, data_path='./data', seed=seed)

trn_val_list = random.sample(trn_list,len(trn_list))[:200]

trn_data = dataset.lonDataList('./data', 'train', trn_list, 224, args.label_type)
tst_data = dataset.lonDataList('./data', 'test', tst_list, 224, args.label_type)
# val_data = dataset.lonDataList('./data', 'test', trn_val_list, 224, args.label_type)
# trn_data = dataset.lonDataEn('./data', 'train', img_size=224, label_type=args.label_type, seed=seed)
# tst_data = dataset.lonDataEn('./data', 'test', img_size=224, label_type=args.label_type, seed=seed)

trn_loader = DataLoader(trn_data, batch_size=batch_size, shuffle=True , num_workers=nworkers, drop_last=True)
tst_loader = DataLoader(tst_data, batch_size=batch_size_val, shuffle=False, num_workers=nworkers)

## ==================================== feature model & longitudinal model ===========================
flag_net = args.feat_net
if flag_net == 'self':
    net_fea = Model.featureNet().to(device)
    num_fits = 2048
    net = Model.downFeaNet(num_fits=2048, feature_size=feature_size).to(device)
    pre_epoch = 0
elif flag_net == 'resnet':
    net = models.resnet50(pretrained=True)
    num_fits = net.fc.in_features
    # net.fc = nn.Linear(num_fits, feature_size)
    feature_size = num_fits
    net.fc = nn.Flatten()
elif flag_net == 'simple':
    net = Model.simpleFeatNet(feature_size=feature_size)
elif flag_net == 'ae':
    load_suffix = '/home/lhq323/Project/phd/zhaoh/Python_scripts/longitudinal/AE_feature/Model_and_Result/first_try/latest_net_G.pth'
    net_fea = Model.define_AE(3, 3, 32, 'simple', 'instance',
                          load_suffix=load_suffix).to(device)
    net = Model.downFeaNet(num_fits=512, feature_size=feature_size).to(device)

    pre_epoch = 0


#
#
# input_data = torch.zeros((1, 3, 224,224))
# net(input_data)

out_len = 1
model = Model.GRUNet(input_dim=feat_len, hidden_dim=hid, output_dim=cls_num, n_layers=nlayers, drop_prob=dropout).to(device)

## =================================== criterion & setting ===========================================

if not args.focal:
    criterion = nn.CrossEntropyLoss()
    criterion_val = nn.CrossEntropyLoss()
else:
    criterion = FocalLoss(alpha=fl_alpha, gamma=fl_gamma)
    # criterion_val = nn.CrossEntropyLoss()
    criterion_val = nn.CrossEntropyLoss()

# optimizer = optim.Adam(model.parameters(), lr=lr)
if args.feat_net == 'self' or args.feat_net == 'ae':
    optimizer = optim.Adam(itertools.chain(net.parameters(),model.parameters(), net_fea.parameters()), lr=lr, betas=(0.5, 0.999))
elif args.feat_net == 'simple':
    optimizer = optim.Adam(itertools.chain(net.parameters(),model.parameters()), lr=lr, betas=(0.5, 0.999))
scheduler = utils.get_scheduler(optimizer, args)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_step, gamma=lr_ratio)

## ======================================= save or load model ===========================================

if load_model:
    path_trans = args.checkpoints_dir + '/' + load_model + '/models/' + 'transNet_0.pth'
    path_fea = args.checkpoints_dir + '/' + load_model + '/models/' + 'feaNet_0.pth'

    model.load_state_dict(torch.load(path_trans))
    net.load_state_dict(torch.load(path_fea))
    print('Loaded saved model')
elif save_model:
    torch.save(model.state_dict(), model_directory + '/transNet_' + str(0) + '.pth')
    # torch.save(net.state_dict(), model_directory + '/feaNet_' + str(0) + '.pth')
    print('Save model done!')

## ======================================= start training ===============================================

# with open(result_dir + '/training_log.txt', 'w') as f:
#     f.write(str(seed))
#     f.close()

best_loss = np.inf
stop_step = 0
best_acc = 0

# pdb.set_trace()

print('Training START !!!')

for epoch in range(epochs):

    lossAll, n_exp = 0, 0
    labelAll, predAll = [], []
    # i=0

    model.train()
    net.train()
    start = time.time()
    #
    if args.feat_net == 'self' or args.feat_net == 'ae':
        if epoch < pre_epoch:
            net_fea.eval()
        else:
            net_fea.train()

    # data = utils.load_var('1.pkl')
    # for i in range(200):
    for i, data in enumerate(trn_loader):
        
        end1 = time.time()
        # print(end1-start)

        lossBlock, b_exp = 0, 0

        # img_path[0][*] and img_path[1][*] are a pair, e.g. 2012_R2.jpg, 2013_R2.jpg
        img_seq, label_seq, img_path, label_path = data

        img_seq, label_seq = img_seq.to(device), label_seq.to(device)
        # merge batch and sequence together [bz, seq_len] -> [bz*seq_len]

        label_seq_vec = label_seq[:, -1]
        # print(label_seq_vec.tolist().count(0), label_seq_vec.tolist().count(1))
        # label = {0, 1, 2}; 0- normal, 1- slightly, 2- seriously

        img_size = img_seq.size()

        # img_seq = torch.reshape(img_seq, (-1, *(img_size[2:])))

        if args.feat_net == 'self':
            if epoch < pre_epoch:
                with torch.no_grad():
                    img_fea = net_fea(img_seq)
            else:
                img_fea = net_fea(img_seq)
            img_fea = net(img_fea)
        elif args.feat_net == 'ae':
            if epoch < pre_epoch:
                with torch.no_grad():
                    img_fea = net_fea(img_seq)['fea']
            else:
                img_fea = net_fea(img_seq)['fea']
            img_fea = net(img_fea)
        else:
            img_fea = net(img_seq)
        #
        # # end2 = time.time()
        #
        # # change to shape [seq_len, bz, fea_len]
        # img_fea = img_fea.reshape(*img_size[:2],-1)


        # if i == 0:
        h = model.init_hidden(batch_size).to(device)
            # h = torch.zeros((2,batch_size,512)).to(device)

        optimizer.zero_grad()
        output_vec, h = model(img_fea, h.data)
        loss = criterion(output_vec, label_seq_vec)

        # loss = criterion(output_vec[:,0], label_seq_vec.type(dtype=torch.float32))
        loss.backward()
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        pred = output_vec.argmax(1).cpu().numpy()
        predAll += pred.tolist()
        labelAll += (label_seq_vec.cpu().numpy()).tolist()

        # if epoch == 0 and i == 0:
        #     loss.backward(retain_graph=True)
        # else:



        # end3 = time.time()

        lossAll += loss.item() * len(label_seq_vec)
        lossBlock += loss.item() * len(label_seq_vec)
        n_exp += len(label_seq_vec)
        b_exp += len(label_seq_vec)

        # print(end2-end1, end3-end2, end1-start)

        if i % int(len(trn_loader) / 5) == 0 and i >0:
            end_time = time.time()
            cur_loss = lossBlock / b_exp
            print('| epoch {:3d} | {:5d}/{:5d} batches, time: {:9.4f} | lr {:02.6f} | loss {:5.4f} | cls_ratio {:4d} / {:4d} / {:4d}'.format(
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

    model.eval()
    net.eval()
    net_fea.eval()

    # data = utils.load_var('2.pkl')
    # for i in range(1):
    for i, data in enumerate(tst_loader):

        total_loss = 0

        img_val, label_val, img_path, label_path = data
        img_val, label_val = img_val.to(device), label_val.to(device)
        label_val_vec = label_val[:,-1]

        img_size = img_val.size()

        h = torch.zeros((nlayers, img_size[0], hid)).to(device)
        with torch.no_grad():

            if args.feat_net == 'self':
                img_fea_val = net_fea(img_val)
                img_fea_val = net(img_fea_val)
            elif args.feat_net == 'ae':
                img_fea_val = net_fea(img_val)['fea']
                img_fea_val = net(img_fea_val)
            else:
                img_fea_val = net(img_val)
            
            output_vec, h = model(img_fea_val, h)

            pred = output_vec.argmax(1).cpu().numpy()
            predAll += pred.tolist()
            # predAll += output_vec[:,0].tolist()

            labelAll += (label_val_vec.cpu().numpy()).tolist()

            loss_val = criterion_val(output_vec, label_val_vec)
            # loss_val = criterion_val(output_vec[:,0], label_val_vec)

            lossAll += loss_val.item() * len(label_val_vec)
            n_exp += len(label_val_vec)


    # pred = np.concatenate(predAll, axis=0)

    # f1, pre, rec, auc, acc, kappa, _, _, _, _ = utils.calc_score(labelAll, predAll)

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
        torch.save(model.state_dict(), model_directory + '/transNet_' + str(epoch+1) + '.pth')
        # torch.save(net.state_dict(), model_directory + '/feaNet_' + str(epoch+1) + '.pth')


    if stop_criteria == 'val_loss':
        if val_loss < best_loss-0.0002:
            best_loss = val_loss
            best_acc = acc
            best_kappa = kappa
            stop_step = 0
            best_epoch = epoch

            torch.save(model.state_dict(), model_directory + '/transNet_best.pth')
            # torch.save(net.state_dict(), model_directory + '/feaNet_best.pth')

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

            torch.save(model.state_dict(), model_directory + '/transNet_best.pth')
            torch.save(net.state_dict(), model_directory + '/feaNet_best.pth')



print('Early stopping is trigger. ----->>>>>> Best loss: {:.4f}, acc: {:.4f}, kappa: {:.4f} at epoch {:3d}'.format(
    best_loss, best_acc, best_kappa, best_epoch+1))
with open(result_dir + '/training_log.txt', 'a') as f:
    f.write('Early stopping is trigger. ----->>>>>> Best loss: {:.4f}, acc: {:.4f}, kappa: {:.4f} at epoch {:3d}'.format(
        best_loss, best_acc, best_kappa, best_epoch+1))


    # # load
    # model.load_state_dict(torch.load(model_directory + '/transNet.pth'))
    # net.load_state_dict(torch.load(model_directory + '/feaNet.pth'))
