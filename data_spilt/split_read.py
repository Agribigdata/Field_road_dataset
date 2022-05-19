import pickle
import os
import shutil



def spilt():
    path = 'wheat_1_data_split.pkl'
    # path = 'paddy_data_split.pkl'
    # path = 'tractor_data_split.pkl'
    # path = 'wheat_1_data_split.pkl'
    # path = 'wheat_2_data_split.pkl'
    Kfold = 10
    with open(path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
        train = data['train']
        valid = data['valid']
        test = data['test']

        #10 fold
        print('train:', len(train))
        print('valid:', len(valid))
        print('test:', len(test))

        for i in range(Kfold):
            fold_path=spilt_path+"/"+str(i)
            if not os.path.exists(fold_path):
                os.mkdir(fold_path)
            if not os.path.exists(fold_path+"/train"):
                os.mkdir(fold_path+"/train")
            if not os.path.exists(fold_path+"/val"):
                os.mkdir(fold_path+"/val")
            if not os.path.exists(fold_path+"/test"):
                os.mkdir(fold_path+"/test")

            for j in train[i]:
                shutil.copyfile(files_path+"/"+j,fold_path+"/train/"+j)
            for j in valid[i]:
                shutil.copyfile(files_path + "/" + j, fold_path + "/val/" + j)
            for j in test[i]:
                shutil.copyfile(files_path + "/" + j, fold_path + "/test/" + j)
            print('len train:',len(train[i]))
            print('train:',train[i])
            print('len valid:',len(valid[i]))
            print('valid:',valid[i])
            print('len test:',len(test[i]))
            print('test:',test[i])
            print('+++++++++++++++++++++++++++++++++++++++++++++++')
if __name__=="__main__":
    files_path = " "#划分数据集
    spilt_path = files_path + "_10fold"
    if not os.path.exists(spilt_path):
        os.mkdir(spilt_path)
    spilt()