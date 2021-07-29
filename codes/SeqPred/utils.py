import argparse
import os
from torch.optim import lr_scheduler
import pickle
from PIL import Image as im
import numpy as np
import torch
import random
import pandas as pd
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, accuracy_score, roc_auc_score, \
    confusion_matrix, cohen_kappa_score

def get_scheduler(optimizer, opt):

    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=opt.lr_ratio)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def print_options(parser, opt):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    mkdirs(expr_dir)
    #import pdb
    #pdb.set_trace()
    file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def save_var(var, path):
    file = open(path, 'wb')
    pickle.dump(var, file)
    file.close()

def load_var(path):
    with open(path, 'rb') as file:
        var = pickle.load(file)

    return var

def save_fig(img, path):
    img1 = im.fromarray(img,'RGB')
    img1.save(path)


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



def calc_score(label, all_score):

    label = np.array(label)
    all_score = np.array(all_score)

    [Pre, Rec, thresholds] = precision_recall_curve(label, all_score)
    F1 = 2 * Pre * Rec / (Pre + Rec)
    thres = thresholds[F1.argmax()]
    labelPred = all_score.copy()
    labelPred[all_score > thres], labelPred[all_score <= thres] = 1, 0
    result = precision_recall_fscore_support(label, labelPred, average='binary')
    acc = accuracy_score(label, labelPred)
    auc = roc_auc_score(label, all_score)
    f1, pre, rec = result[2], result[0], result[1]
    tn, fp, fn, tp = confusion_matrix(label, labelPred).ravel()
    kappa = cohen_kappa_score(label, labelPred)

    return f1, pre, rec, auc, acc, kappa, tn, fp, fn, tp

def idList(label_type, data_path, seed):


    if label_type == 'diag':
        csv_file = data_path + '/' + 'usedID.csv'
    elif label_type == 'pre_diabet':
        csv_file = data_path + '/' + 'usedID_prediabetes.csv'
    elif label_type == 'pre_diabet_2':
        csv_file = data_path + '/' + 'usedID_prediabetes.csv'
    else:
        csv_file = data_path + '/' + 'usedID_glucose.csv'


    dfname = pd.read_csv(csv_file)
    tjid_all = list(dfname.tjid)

    # seed = random.randint(1,5000)
    random.seed(seed)
    tst_list = random.sample(tjid_all, 200)
    trn_list = list(set(tjid_all) - set(tst_list))

    return trn_list, tst_list







