import torch
from torch.utils.data import DataLoader
from dataset import lonData
import torch.nn as nn
from torchvision import models
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score, f1_score
from Model import TransImgPred


device = torch.device('cuda:0')

exp_name = 'diabet_2cls'
epoch = '80'

feature_merge = False
single_pred = True

label_type = 'pre_diabet_2'

if label_type == 'pre_diabet':
    cls_num = 3
elif label_type == 'pre_diabet_2':
    cls_num = 2
elif label_type == 'diag':
    cls_num = 2    


batch_size = 32
nworkers = 8
feature_size = 256
img_len = 2
dropout = 0


result_dir = 'Model_and_Result' + '/' + exp_name
model_directory = result_dir + '/models'

tst_data = lonData('./data', 'test', img_size=224, label_type=label_type)
tst_loader = DataLoader(tst_data, batch_size=batch_size, shuffle=False, num_workers=nworkers)


## ====================== model ===============================
net = models.resnet50(pretrained=True)
num_fits = net.fc.in_features
# net.fc = nn.Linear(num_fits, feature_size)
feature_size = num_fits
net.fc = nn.Flatten()
net = net.to(device)
net.eval()


if feature_merge == True:
    model = TransImgPred(seq_len=img_len, feature_size=feature_size, cls_num=cls_num, nhid=512,
                         nlayers=6, nhead=8, dropout=dropout, merge=True).to(device)
    out_len = 1
else:
    model = TransImgPred(seq_len=img_len, feature_size=feature_size, cls_num=cls_num, nhid=512,
                         nlayers=6, nhead=8, dropout=dropout).to(device)
    out_len = 2

model.load_state_dict(torch.load(model_directory + '/transNet_' + epoch + '.pth'))
net.load_state_dict(torch.load(model_directory + '/feaNet_' + epoch + '.pth'))


predAll, labelAll = [], []

model.eval()
for i, data in enumerate(tst_loader):

    total_loss = 0

    base_val, img_val, label_val, img_path, label_path = data
    base_val, img_val, label_val = base_val.to(device), img_val.to(device), label_val.to(device)
    label_val_vec = label_val.permute(1,0).reshape((label_val.size(0)*img_len))
    
    if feature_merge or single_pred == True:
        label_val_vec = label_val_vec[-label_val.size(0):]

    base_size, img_size = base_val.size(), img_val.size()

    tgt_mask = model.generate_square_subsequent_mask(img_size[1]).to(device)



    with torch.no_grad():

        base_val = torch.reshape(base_val, (-1, *(base_size[2:])))
        img_val = torch.reshape(img_val, (-1, *(img_size[2:])))

        base_fea_val = net(base_val)
        img_fea_val = net(img_val)

        base_fea_val = base_fea_val.reshape(*base_size[:2],-1).permute(1,0,2)
        img_fea_val = img_fea_val.reshape(*img_size[:2],-1).permute(1,0,2)

        output, _ = model(base_fea_val, img_fea_val, tgt_mask=tgt_mask)
        output_vec = output.reshape((img_size[0]*out_len, -1))

        if single_pred == True:
            output_vec = output_vec[-label_val.size(0):]

        pred = output_vec.argmax(1).cpu().numpy()
        predAll += pred.tolist()
        labelAll += (label_val_vec.cpu().numpy()).tolist()


acc = accuracy_score(labelAll, predAll)
kappa = cohen_kappa_score(labelAll, predAll)
f1 = f1_score(labelAll, predAll, average='macro')

print('acc {:5.4f}, kappa {:5.4f}, F1 {:5.4f} | cls_ratio {:4d} / {:4d} / {:4d}'.format(
     acc, kappa, f1, labelAll.count(0), labelAll.count(1), labelAll.count(2)))
