
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
import copy
import random
import glob
from sklearn.model_selection import train_test_split

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
parser.add_argument('--batch_size', default=8, type=int)

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

feature_size = 1024
feat_len = feature_size #* 7 * 7
img_len = 49
nhead = 4
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

seed = random.randint(1,50000)
seed = 2021
print(seed)

train_dir = 'files/dog_cat/train'
test_dir = 'files/dog_cat/test'
train_list = glob.glob(os.path.join(train_dir,'*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))
labels = [path.split('/')[-1].split('.')[0] for path in train_list]
train_list, valid_list = train_test_split(train_list, 
                                          test_size=0.2,
                                          stratify=labels,
                                          random_state=seed)

from torchvision import datasets, transforms
train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)


test_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm as tqdm
class CatsDogsDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split("/")[-1].split(".")[0]
        label = 1 if label == "dog" else 0

        return img_transformed, label


train_data = CatsDogsDataset(train_list, transform=train_transforms)
valid_data = CatsDogsDataset(valid_list, transform=test_transforms)
test_data = CatsDogsDataset(test_list, transform=test_transforms)

train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)

trn_loader = train_loader

tst_loader = valid_loader

## ==================================== feature model & longitudinal model ===========================
flag_net = args.feat_net
if flag_net == 'self':
    net_fea = Model.featureNet()
    net_fea.to(device)
    num_fits = 2048
    net = Model.downFeaNet_TPT(num_fits=2048, feature_size=128)
    pre_epoch = 0
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


model = Model.TPT(seq_len=img_len, feature_size=feat_len, featMap_size=1, num_classes=cls_num, depth=nlayers, heads=nhead,
                      mlp_dim=128, pool='cls',channels=128,dim_head=64,dropout=dropout).to(device)                   
out_len = 1


## =================================== criterion & setting ===========================================

if not args.focal:
    criterion = nn.CrossEntropyLoss()
    criterion_val = nn.CrossEntropyLoss()
else:
    criterion = FocalLoss(alpha=fl_alpha, gamma=fl_gamma)
    # criterion_val = nn.CrossEntropyLoss()
    criterion_val = nn.CrossEntropyLoss()

# optimizer = optim.Adam(model.parameters(), lr=lr)
if args.feat_net == 'self':
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
    torch.save(net.state_dict(), model_directory + '/feaNet_' + str(0) + '.pth')
    print('Save model done!')

## ======================================= start training ===============================================

with open(result_dir + '/training_log.txt', 'w') as f:
    f.close()

best_loss = np.inf
stop_step = 0
best_acc = 0

# pdb.set_trace()

print('Training START !!!')


for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    model.train()
    net_fea.train()
    net.train()
    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        fea = net_fea(data)
        fea = net(fea)
        output = model(fea)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    model.eval()
    net_fea.eval()
    net.eval()

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)

            fea = net_fea(data)
            fea = net(fea)
            val_output = model(fea)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )


for epoch in range(epochs):

    lossAll, n_exp = 0, 0
    labelAll, predAll = [], []
    # i=0

    model.train()
    net.train()
    start = time.time()

    if args.feat_net == 'self':
        if epoch < pre_epoch :
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
        img_seq_base, label_seq, img_path, label_path = data
        img_seq_base, label_seq = img_seq_base.to(device), label_seq.to(device)
        # merge batch and sequence together [bz, seq_len] -> [bz*seq_len]

        img_seq2 = copy.deepcopy(img_seq_base)[:,1:]
        img_seq1 = copy.deepcopy(img_seq_base)[:,:-1]
        img_seq = torch.cat((img_seq1, img_seq2), dim=1)

        label_seq_vec = label_seq[:,5]
        # label = {0, 1, 2}; 0- normal, 1- slightly, 2- seriously

        img_size = img_seq.size()

        img_seq = img_seq.view(-1, *(img_size[2:]))

        if args.feat_net == 'self':
            if epoch < pre_epoch :
                with torch.no_grad():
                    img_fea = net_fea(img_seq)
            else:
                img_fea = net_fea(img_seq)
            img_fea = net(img_fea)
        else:
            img_fea = net(img_seq)

        # end2 = time.time()

        # change to shape [seq_len, bz, fea_len]
        # img_fea = img_fea.reshape(*img_size[:2],-1).permute(1,0,2)
        img_fea = img_fea.view((*img_size[:2],*img_fea.size()[1:]))
        # img_fea1 = img_fea[:,:5].reshape((-1, *img_fea.size()[2:]))
        # img_fea2 = img_fea[:,5:].reshape((-1, *img_fea.size()[2:]))
        # img_fea = torch.subtract(img_fea2,img_fea1)
        img_fea1 = img_fea[:,:5]
        img_fea2 = img_fea[:,5:]
        img_fea = torch.subtract(img_fea2,img_fea1)


        optimizer.zero_grad()

        output_vec = model(img_fea)
        loss = criterion(output_vec, label_seq_vec)

        pred = output_vec.argmax(1).cpu().numpy()
        predAll += pred.tolist()
        labelAll += (label_seq_vec.cpu().numpy()).tolist()

        loss.backward()
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
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
    disAll = []

    model.eval()
    net.eval()
    if args.feat_net == 'self':
        net_fea.eval()

    # data = utils.load_var('2.pkl')
    # for i in range(1):
    for i, data in enumerate(tst_loader):

        total_loss = 0

        img_val, label_val, img_path, label_path = data
        img_val, label_val = img_val.to(device), label_val.to(device)
        label_val_vec = label_val[:,5]
        
        img_size = img_val.size()

        img_val = img_val.view(-1, *(img_size[2:]))

        with torch.no_grad():
            
            if args.feat_net == 'self':
                img_fea_val = net_fea(img_val)
                img_fea_val = net(img_fea_val)
            else:
                img_fea_val = net(img_val)

            img_fea_val = img_fea_val.view((*img_size[:2],*img_fea_val.size()[1:]))

            output_vec = model(img_fea_val)
            disease = torch.nn.Softmax(dim=1)(output_vec)[:,1].cpu().numpy()

            pred = output_vec.argmax(1).cpu().numpy()
            predAll += pred.tolist()
            labelAll += (label_val_vec.cpu().numpy()).tolist()
            disAll += disease.tolist()

            loss_val = criterion_val(output_vec, label_val_vec)

            lossAll += loss_val.item() * len(label_val_vec)
            n_exp += len(label_val_vec)


    # pred = np.concatenate(predAll, axis=0)

    auc = roc_auc_score(labelAll, disAll)
    acc = accuracy_score(labelAll, predAll)
    kappa = cohen_kappa_score(labelAll, predAll)
    f1 = f1_score(labelAll, predAll, average='macro')
    val_loss = lossAll / n_exp

    end_time = time.time()
    print('| epoch {:3d} | time {:5.4f} | train_loss {:5.4f}, val_loss {:5.4f} | acc {:5.4f}, kappa {:5.4f}, F1 {:5.4f}, auc {:5.4f} | cls_ratio {:4d} / {:4d} / {:4d}'.format(
        epoch+1, end_time-start, train_loss, val_loss, acc, kappa, f1, auc, labelAll.count(0), labelAll.count(1), labelAll.count(2)))
    with open(result_dir + '/training_log.txt', 'a') as f:
        f.write('| epoch {:3d} | time {:5.4f} | train_loss {:5.4f}, val_loss {:5.4f}| acc {:5.4f}, kappa {:5.4f}, F1 {:5.4f} | cls_ratio {:4d} / {:4d} / {:4d} \n'.format(
            epoch+1, end_time-start, train_loss, val_loss, acc, kappa, f1, labelAll.count(0), labelAll.count(1), labelAll.count(2)))

    if (epoch+1) % 20 == 0:
        torch.save(model.state_dict(), model_directory + '/transNet_' + str(epoch+1) + '.pth')
        torch.save(net.state_dict(), model_directory + '/feaNet_' + str(epoch+1) + '.pth')


    if stop_criteria == 'val_loss':
        if val_loss < best_loss-0.0002:
            best_loss = val_loss
            best_acc = acc
            best_kappa = kappa
            stop_step = 0
            best_epoch = epoch

            torch.save(model.state_dict(), model_directory + '/transNet_best.pth')
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
