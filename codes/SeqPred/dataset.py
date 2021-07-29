import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms
import numpy as np
import random

class lonData(Dataset):
    def __init__(self, data_path, state, img_size, label_type='pre_diabet'):
        self.data_path = data_path
        self.img_size = img_size
        self.state = state

        if state == 'train':
            flag = 'trn'
        elif state == 'test':
            flag = 'tst'


        ## type: diag, pre_diabet
        self.label_type = label_type

        if self.label_type == 'diag':
            self.csv_file = data_path + '/' + flag + '_id.csv'
            df = pd.read_csv(data_path + '/train_clear.csv')
        elif self.label_type == 'pre_diabet':
            self.csv_file = data_path + '/' + flag + '_id_pre.csv'
            df = pd.read_csv(data_path + '/tjid_Pre-diabetes_0420.csv')
        elif self.label_type == 'pre_diabet_2':
            self.csv_file = data_path + '/' + flag + '_id_pre.csv'
            df = pd.read_csv(data_path + '/tjid_Pre-diabetes_0420.csv')

        dfname = pd.read_csv(self.csv_file)
        self.file_list = list(dfname.tjid)
        self.label = df[['tjid','diabetes_2010', 'diabetes_2011','diabetes_2012','diabetes_2013','diabetes_2014','diabetes_2015','diabetes_2016']]

        self.totensor = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])



        self.seq_num = 3


    def __getitem__(self, item):
        baseList = [2010, 2011]

        seq_num = self.seq_num

        if item % seq_num == 0:
            imgList = [2014, 2015]
            labelList = [2015, 2016]
        elif item % seq_num == 1:
            imgList = [2013, 2014]
            labelList = [2014, 2015]
        elif item % seq_num == 2:
            imgList = [2012, 2013]
            labelList = [2013, 2014]
        elif item % seq_num == 3:
            imgList = [2011, 2012]
            labelList = [2012, 2013]


        import time
        # start = time.time()

        item_id = item%(len(self.file_list))
        tjid = self.file_list[item_id]


        base_seq, base_path = [], []
        img_seq, img_path = [], []
        label_seq, label_path = [], []

        start=time.time()
        for i in baseList:
            image_path = self.data_path + '/idFile/' + str(tjid) + '/' + str(tjid) + '_' + str(i) + '_R2.jpg'
            img = Image.open(image_path)
            base_seq.append(self.totensor(img))
            base_path.append(image_path)

        base_seq = torch.stack(base_seq)

        # end1 = time.time()

        for i in imgList:
            image_path = self.data_path + '/idFile/' + str(tjid) + '/' + str(tjid) + '_' + str(i) + '_R2.jpg'
            img = Image.open(image_path)
            img_seq.append(self.totensor(img))
            img_path.append(image_path)

        img_seq = torch.stack(img_seq)

        # end2 = time.time()

        for i in labelList:
            # label = self.label['diabetes_' + str(i)][item_id]
            label = self.label[self.label.tjid==tjid]['diabetes_' + str(i)].item()

            if self.label_type == 'diag':
                if label == 'no':
                    label = 0
                elif label == 'yes':
                    label = 1
            elif self.label_type == 'pre_diabet':
                if label == 'Normal':
                    label = 0
                elif label == 'Pre-diabetes':
                    label = 1
                elif label == 'Diabetes':
                    label = 2
            elif self.label_type == 'pre_diabet_2':
                if label == 'Normal':
                    label = 0
                elif label == 'Pre-diabetes':
                    label = 1
                elif label == 'Diabetes':
                    label = 1

            label_seq.append(label)
            label_path.append(self.data_path + '/idFile/' + str(tjid) + '/' + str(tjid) + '_' + str(i) + '_R2.jpg')
        
        label_seq = torch.from_numpy(np.stack(label_seq))

        # end3 = time.time()
        # print(end1-start, end2-end1, end3-end2)

        return base_seq, img_seq, label_seq, img_path, label_path


    def __getitem1__(self, item):
        baseList = [2010, 2011]

        seq_num = self.seq_num

        if item % seq_num == 0:
            imgList = [2014, 2015]
            labelList = [2015, 2016]
        elif item % seq_num == 1:
            imgList = [2013, 2014]
            labelList = [2014, 2015]
        elif item % seq_num == 2:
            imgList = [2012, 2013]
            labelList = [2013, 2014]
        elif item % seq_num == 3:
            imgList = [2011, 2012]
            labelList = [2012, 2013]


        import time
        # start = time.time()

        item_id = item%(len(self.file_list))
        tjid = self.file_list[item_id]


        base_seq, base_path = [], []
        img_seq, img_path = [], []
        label_seq, label_path = [], []

        start=time.time()
        for i in baseList:
            image_path = self.data_path + '/idFile/' + str(tjid) + '/' + str(tjid) + '_' + str(i) + '_R2.jpg'
            img = self.image_read(image_path, self.img_size)
            end1 =time.time()
            base_seq.append(img)
            end2 = time.time()
            base_path.append(image_path)
        end3 = time.time()
        base_seq = np.stack(base_seq)
        end4 = time.time()

        # end1 = time.time()

        for i in imgList:
            image_path = self.data_path + '/idFile/' + str(tjid) + '/' + str(tjid) + '_' + str(i) + '_R2.jpg'
            img = Image.open(image_path)
            img_seq.append(self.totensor(img))
            img_path.append(image_path)

        img_seq = torch.stack(img_seq)

        # end2 = time.time()

        for i in labelList:
            # label = self.label['diabetes_' + str(i)][item_id]
            label = self.label[self.label.tjid==tjid]['diabetes_' + str(i)].item()

            if self.label_type == 'diag':
                if label == 'no':
                    label = 0
                elif label == 'yes':
                    label = 1
            elif self.label_type == 'pre_diabet':
                if label == 'Normal':
                    label = 0
                elif label == 'Pre-diabetes':
                    label = 1
                elif label == 'Diabetes':
                    label = 2

            label_seq.append(label)
            label_path.append(self.data_path + '/idFile/' + str(tjid) + '/' + str(tjid) + '_' + str(i) + '_R2.jpg')

        label_seq = torch.from_numpy(np.stack(label_seq))

        # end3 = time.time()
        # print(end1-start, end2-end1, end3-end2)

        return base_seq, img_seq, label_seq, img_path, label_path

    def __len__(self):

        return(len(self.file_list)*self.seq_num)


class lonDataEn(Dataset):
    def __init__(self, data_path, state, img_size, seed=2021, label_type='pre_diabet'):
        self.data_path = data_path
        self.img_size = img_size
        self.state = state

        ## type: diag, pre_diabet
        self.label_type = label_type

        if self.label_type == 'diag':
            # self.csv_file = data_path + '/' + flag + '_id.csv'    
            self.csv_file = data_path + '/' + 'usedID.csv'        
            df = pd.read_csv(data_path + '/train_clear.csv')
        elif self.label_type == 'pre_diabet':
            # self.csv_file = data_path + '/' + flag + '_id_pre.csv'
            self.csv_file = data_path + '/' + 'usedID_prediabetes.csv'
            df = pd.read_csv(data_path + '/tjid_Pre-diabetes_0420.csv')
        elif self.label_type == 'pre_diabet_2':
            # self.csv_file = data_path + '/' + flag + '_id_pre.csv'
            self.csv_file = data_path + '/' + 'usedID_prediabetes.csv'
            df = pd.read_csv(data_path + '/tjid_Pre-diabetes_0420.csv')

        dfname = pd.read_csv(self.csv_file)
        tjid_all = list(dfname.tjid)
        self.label = df[['tjid','diabetes_2010', 'diabetes_2011','diabetes_2012','diabetes_2013','diabetes_2014','diabetes_2015','diabetes_2016']]

        if state == 'train':
            self.totensor = transforms.Compose([
                transforms.Resize((img_size,img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=180),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        elif state == 'test':
            self.totensor = transforms.Compose([
                transforms.Resize((img_size,img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        
        random.seed(seed)
        tst_list = random.sample(tjid_all, 200)
        trn_list = list(set(tjid_all)-set(tst_list))


        if state == 'train':
            self.file_list = trn_list
        elif state == 'test':
            self.file_list = tst_list


        self.seq_num = 3


    def __getitem__(self, item):
        baseList = [2010, 2011, 2012, 2013, 2014, 2015]
        labelList = [2011, 2012, 2013, 2014, 2015, 2016]

        item_id = item
        tjid = self.file_list[item_id]


        base_seq, base_path = [], []
        label_seq, label_path = [], []

        for i in baseList:
            image_path = self.data_path + '/idFile/' + str(tjid) + '/' + str(tjid) + '_' + str(i) + '_R2.jpg'
            img = Image.open(image_path)
            base_seq.append(self.totensor(img))
            base_path.append(image_path)

        img_path = base_path
        img_seq = torch.stack(base_seq)

        for i in labelList:
            # label = self.label['diabetes_' + str(i)][item_id]
            label = self.label[self.label.tjid==tjid]['diabetes_' + str(i)].item()

            if self.label_type == 'diag':
                if label == 'no':
                    label = 0
                elif label == 'yes':
                    label = 1
            elif self.label_type == 'pre_diabet':
                if label == 'Normal':
                    label = 0
                elif label == 'Pre-diabetes':
                    label = 1
                elif label == 'Diabetes':
                    label = 2
            elif self.label_type == 'pre_diabet_2':
                if label == 'Normal':
                    label = 0
                elif label == 'Pre-diabetes':
                    label = 1
                elif label == 'Diabetes':
                    label = 1

            label_seq.append(label)
            label_path.append(self.data_path + '/idFile/' + str(tjid) + '/' + str(tjid) + '_' + str(i) + '_R1.jpg')
        
        label_seq = torch.from_numpy(np.stack(label_seq))

        # end3 = time.time()
        # print(end1-start, end2-end1, end3-end2)

        return img_seq, label_seq, img_path, label_path
    
    def __len__(self):

        return len(self.file_list)



class lonDataList(Dataset):
    def __init__(self, data_path, state, file_list, img_size, label_type='pre_diabet'):
        self.data_path = data_path
        self.img_size = img_size
        self.state = state

        ## type: diag, pre_diabet
        self.label_type = label_type

        if self.label_type == 'diag':
            df = pd.read_csv(data_path + '/label_all/train_clear.csv')
            self.label = df[['tjid','diabetes_2010', 'diabetes_2011','diabetes_2012','diabetes_2013','diabetes_2014','diabetes_2015','diabetes_2016']]

        elif self.label_type == 'pre_diabet':
            df = pd.read_csv(data_path + '/label_all/tjid_Pre-diabetes_0420.csv')
            self.label = df[['tjid','diabetes_2010', 'diabetes_2011','diabetes_2012','diabetes_2013','diabetes_2014','diabetes_2015','diabetes_2016']]

        elif self.label_type == 'pre_diabet_2':
            df = pd.read_csv(data_path + '/label_all/tjid_Pre-diabetes_0420.csv')
            self.label = df[['tjid','diabetes_2010', 'diabetes_2011','diabetes_2012','diabetes_2013','diabetes_2014','diabetes_2015','diabetes_2016']]

        elif self.label_type == 'glucose':
            df = pd.read_csv(data_path + '/label_all/GlucoseChange_Total.csv')
            self.label = df[['tjid','Glucose_Change', 'CurrentYear', 'LastYear']]


        # self.label = df[['tjid','diabetes_2010', 'diabetes_2011','diabetes_2012','diabetes_2013','diabetes_2014','diabetes_2015','diabetes_2016']]

        if state == 'train':
            self.totensor = transforms.Compose([
                transforms.Resize((img_size,img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=180),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        elif state == 'test':
            self.totensor = transforms.Compose([
                transforms.Resize((img_size,img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        self.file_list = file_list

        self.seq_num = 3

    def get_label(self, tjid, i):
        label = self.label[self.label.tjid==tjid]['diabetes_' + str(i)].item()

        return label

    def get_label_glucose(self, tjid, i):

        try:
            label = self.label.loc[self.label.tjid==tjid].loc[self.label.CurrentYear==i]['Glucose_Change'].item()
        except:
            label = 'Increase'
            print(str(tjid))

        return label


    def __getitem__(self, item):
        baseList = [2010, 2011, 2012, 2013, 2014, 2015]
        labelList = [2011, 2012, 2013, 2014, 2015, 2016]

        item_id = item
        tjid = self.file_list[item_id]


        base_seq, base_path = [], []
        label_seq, label_path = [], []

        for i in baseList:
            image_path = self.data_path + '/idFile/' + str(tjid) + '/' + str(tjid) + '_' + str(i) + '_R2.jpg'
            img = Image.open(image_path)
            base_seq.append(self.totensor(img))
            base_path.append(image_path)

        img_path = base_path
        img_seq = torch.stack(base_seq)

        for i in labelList:
            # label = self.label['diabetes_' + str(i)][item_id]
            # label = self.label[self.label.tjid==tjid]['diabetes_' + str(i)].item()


            if self.label_type == 'diag':
                label = self.get_label(tjid, i)
                if label == 'no':
                    label = 0
                elif label == 'yes':
                    label = 1
            elif self.label_type == 'pre_diabet':
                label = self.get_label(tjid, i)
                if label == 'Normal':
                    label = 0
                elif label == 'Pre-diabetes':
                    label = 1
                elif label == 'Diabetes':
                    label = 2
            elif self.label_type == 'pre_diabet_2':
                label = self.get_label(tjid, i)
                if label == 'Normal':
                    label = 0
                elif label == 'Pre-diabetes':
                    label = 1
                elif label == 'Diabetes':
                    label = 1
            elif self.label_type == 'glucose':
                label = self.get_label_glucose(tjid, i)
                if label == 'Decrease':
                    label = 0
                elif label == 'No Change':
                    label = 1
                elif label == 'Increase':
                    label = 2


            label_seq.append(label)
            label_path.append(self.data_path + '/idFile/' + str(tjid) + '/' + str(tjid) + '_' + str(i) + '_R1.jpg')

        label_seq = torch.from_numpy(np.stack(label_seq))

        # end3 = time.time()
        # print(end1-start, end2-end1, end3-end2)

        return img_seq, label_seq, img_path, label_path

    def __len__(self):

        return len(self.file_list)




class lonDataEn_min(Dataset):
    def __init__(self, data_path, state, img_size, label_type='pre_diabet'):
        self.data_path = data_path
        self.img_size = img_size
        self.state = state

        if state == 'train':
            flag = 'trn'
        elif state == 'test':
            flag = 'tst'


        ## type: diag, pre_diabet
        self.label_type = label_type

        if self.label_type == 'diag':
            self.csv_file = data_path + '/' + flag + '_id.csv'
            df = pd.read_csv(data_path + '/train_clear.csv')
        elif self.label_type == 'pre_diabet':
            self.csv_file = data_path + '/' + flag + '_id_pre.csv'
            df = pd.read_csv(data_path + '/tjid_Pre-diabetes_0420.csv')
        elif self.label_type == 'pre_diabet_2':
            self.csv_file = data_path + '/' + flag + '_id_pre.csv'
            df = pd.read_csv(data_path + '/tjid_Pre-diabetes_0420.csv')

        dfname = pd.read_csv(self.csv_file)
        self.file_list = list(dfname.tjid)
        self.label = df[['tjid','diabetes_2010', 'diabetes_2011','diabetes_2012','diabetes_2013','diabetes_2014','diabetes_2015','diabetes_2016']]

        self.totensor = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
        ])



        self.seq_num = 3


    def __getitem__(self, item):
        baseList = [2010, 2011, 2012, 2013, 2014, 2015]
        labelList = [2011, 2012, 2013, 2014, 2015, 2016]

        item_id = item
        tjid = self.file_list[item_id]


        base_seq, base_path = [], []
        label_seq, label_path = [], []

        for i in baseList:
            image_path = self.data_path + '/idFile/' + str(tjid) + '/' + str(tjid) + '_' + str(i) + '_R2.jpg'
            img = Image.open(image_path)
            base_seq.append(self.totensor(img))
            base_path.append(image_path)

        img_path = base_path
        img_seq = torch.stack(base_seq)

        for i in labelList:
            # label = self.label['diabetes_' + str(i)][item_id]
            label = self.label[self.label.tjid==tjid]['diabetes_' + str(i)].item()

            if self.label_type == 'diag':
                if label == 'no':
                    label = 0
                elif label == 'yes':
                    label = 1
            elif self.label_type == 'pre_diabet':
                if label == 'Normal':
                    label = 0
                elif label == 'Pre-diabetes':
                    label = 1
                elif label == 'Diabetes':
                    label = 2
            elif self.label_type == 'pre_diabet_2':
                if label == 'Normal':
                    label = 0
                elif label == 'Pre-diabetes':
                    label = 1
                elif label == 'Diabetes':
                    label = 1

            label_seq.append(label)
            label_path.append(self.data_path + '/idFile/' + str(tjid) + '/' + str(tjid) + '_' + str(i) + '_R2.jpg')
        
        label_seq = torch.from_numpy(np.stack(label_seq))

        # end3 = time.time()
        # print(end1-start, end2-end1, end3-end2)

        return img_seq, label_seq, img_path, label_path
    
    def __len__(self):

        return len(self.file_list)




class lonData_resnet(Dataset):
    def __init__(self, data_path, state, img_size, label_type='diag'):
        self.data_path = data_path
        self.img_size = img_size
        self.state = state

        if state == 'train':
            flag = 'trn'
        elif state == 'test':
            flag = 'tst'


        ## type: diag, pre_diabet
        self.label_type = label_type

        if self.label_type == 'diag':
            self.csv_file = data_path + '/' + flag + '_id.csv'
            df = pd.read_csv(data_path + '/train_clear.csv')


        dfname = pd.read_csv(self.csv_file)
        self.file_list = list(dfname.tjid)
        self.label = df[['tjid','diabetes_2010', 'diabetes_2011','diabetes_2012','diabetes_2013','diabetes_2014','diabetes_2015','diabetes_2016']]

        self.totensor = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])



        self.seq_num = 3


    def __getitem__(self, item):
        baseList = [2010, 2011]

        seq_num = self.seq_num

        if item % seq_num == 0:
            imgList = [2014, 2015]
            labelList = [2015, 2016]
        elif item % seq_num == 1:
            imgList = [2013, 2014]
            labelList = [2014, 2015]
        elif item % seq_num == 2:
            imgList = [2012, 2013]
            labelList = [2013, 2014]
        elif item % seq_num == 3:
            imgList = [2011, 2012]
            labelList = [2012, 2013]


        import time
        # start = time.time()

        item_id = item%(len(self.file_list))
        tjid = self.file_list[item_id]


        base_seq, base_path = [], []
        img_seq, img_path = [], []
        label_seq, label_path = [], []

        start=time.time()
        for i in baseList:
            image_path = self.data_path + '/idFile/' + str(tjid) + '/' + str(tjid) + '_' + str(i) + '_R2.jpg'
            img = Image.open(image_path)
            base_seq.append(self.totensor(img))
            base_path.append(image_path)

        base_seq = torch.stack(base_seq)

        # end1 = time.time()

        for i in imgList:
            image_path = self.data_path + '/idFile/' + str(tjid) + '/' + str(tjid) + '_' + str(i) + '_R2.jpg'
            img = Image.open(image_path)
            img_seq.append(self.totensor(img))
            img_path.append(image_path)

        img_seq = torch.stack(img_seq)

        # end2 = time.time()

        for i in labelList:
            # label = self.label['diabetes_' + str(i)][item_id]
            label = self.label[self.label.tjid==tjid]['diabetes_' + str(i)].item()

            if self.label_type == 'diag':
                if label == 'no':
                    label = 0
                elif label == 'yes':
                    label = 1
            elif self.label_type == 'pre_diabet':
                if label == 'Normal':
                    label = 0
                elif label == 'Pre-diabetes':
                    label = 1
                elif label == 'Diabetes':
                    label = 2
            elif self.label_type == 'pre_diabet_2':
                if label == 'Normal':
                    label = 0
                elif label == 'Pre-diabetes':
                    label = 1
                elif label == 'Diabetes':
                    label = 1

            label_seq.append(label)
            label_path.append(self.data_path + '/idFile/' + str(tjid) + '/' + str(tjid) + '_' + str(i) + '_R2.jpg')
        
        label_seq = torch.from_numpy(np.stack(label_seq))

        # end3 = time.time()
        # print(end1-start, end2-end1, end3-end2)

        return base_seq, img_seq, label_seq, img_path, label_path


    def __getitem1__(self, item):
        baseList = [2010, 2011]

        seq_num = self.seq_num

        if item % seq_num == 0:
            imgList = [2014, 2015]
            labelList = [2015, 2016]
        elif item % seq_num == 1:
            imgList = [2013, 2014]
            labelList = [2014, 2015]
        elif item % seq_num == 2:
            imgList = [2012, 2013]
            labelList = [2013, 2014]
        elif item % seq_num == 3:
            imgList = [2011, 2012]
            labelList = [2012, 2013]


        import time
        # start = time.time()

        item_id = item%(len(self.file_list))
        tjid = self.file_list[item_id]


        base_seq, base_path = [], []
        img_seq, img_path = [], []
        label_seq, label_path = [], []

        start=time.time()
        for i in baseList:
            image_path = self.data_path + '/idFile/' + str(tjid) + '/' + str(tjid) + '_' + str(i) + '_R2.jpg'
            img = self.image_read(image_path, self.img_size)
            end1 =time.time()
            base_seq.append(img)
            end2 = time.time()
            base_path.append(image_path)
        end3 = time.time()
        base_seq = np.stack(base_seq)
        end4 = time.time()

        # end1 = time.time()

        for i in imgList:
            image_path = self.data_path + '/idFile/' + str(tjid) + '/' + str(tjid) + '_' + str(i) + '_R2.jpg'
            img = Image.open(image_path)
            img_seq.append(self.totensor(img))
            img_path.append(image_path)

        img_seq = torch.stack(img_seq)

        # end2 = time.time()

        for i in labelList:
            # label = self.label['diabetes_' + str(i)][item_id]
            label = self.label[self.label.tjid==tjid]['diabetes_' + str(i)].item()

            if self.label_type == 'diag':
                if label == 'no':
                    label = 0
                elif label == 'yes':
                    label = 1
            elif self.label_type == 'pre_diabet':
                if label == 'Normal':
                    label = 0
                elif label == 'Pre-diabetes':
                    label = 1
                elif label == 'Diabetes':
                    label = 2

            label_seq.append(label)
            label_path.append(self.data_path + '/idFile/' + str(tjid) + '/' + str(tjid) + '_' + str(i) + '_R2.jpg')

        label_seq = torch.from_numpy(np.stack(label_seq))

        # end3 = time.time()
        # print(end1-start, end2-end1, end3-end2)

        return base_seq, img_seq, label_seq, img_path, label_path

    def __len__(self):

        return(len(self.file_list)*self.seq_num)





if __name__ == '__main__':

    ass = lonData('data','train',2)