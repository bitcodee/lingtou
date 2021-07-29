import os
import pandas as pd
import shutil

dataPath = './data/washed/'
resultPath = './data/idFile/'

df = pd.read_csv('./data/tjid+diabetes_20210223.csv')
ids = df['tjid'][~df['tjid'].duplicated()].values

txtfile = resultPath + 'missingList.txt'
with open(txtfile, 'w') as f:
    f.close()

for year in range(2009, 2010):
    print(str(year) + 'Start!')
    fileDir = dataPath + str(year) + '/'

    for i, id in enumerate(ids):
        resultDir = resultPath + str(id) + '/'
        if not os.path.isdir(resultDir):
            os.makedirs(resultDir)

        try:
            shutil.copy(fileDir + str(id) + '_L1.jpg', resultDir + str(id) + '_' + str(year) + '_L1.jpg')
        except:
            with open(txtfile, 'a') as f:
                f.write(str(id) + ',' + '_L1' + '\n')

        try:
            shutil.copy(fileDir + str(id) + '_L2.jpg', resultDir + str(id) + '_' + str(year) + '_L2.jpg')
        except:
            with open(txtfile, 'a') as f:
                f.write(str(id) + ',' + '_L2' + '\n')

        try:
            shutil.copy(fileDir + str(id) + '_R1.jpg', resultDir + str(id) + '_' + str(year) + '_R1.jpg')
        except:
            with open(txtfile, 'a') as f:
                f.write(str(id) + ',' + '_R1' + '\n')

        try:
            shutil.copy(fileDir + str(id) + '_R2.jpg', resultDir + str(id) + '_' + str(year) + '_R2.jpg')
        except:
            with open(txtfile, 'a') as f:
                f.write(str(id) + ',' + '_R2' + '\n')

        print('[' +  str(i+1) + '/' + str(len(ids)) + '] ' + str(id) + ' done!')




