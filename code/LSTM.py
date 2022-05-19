
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
from modules.model import LSTM
from torch import optim
import numpy as np
from numpy import mean
from sklearn.metrics import classification_report,accuracy_score
import pandas as pd
import warnings
from dataset_train.data_split import spilt_data
import shutil
warnings.filterwarnings("ignore")
batch_size = 128
input_size = 8 #input_size / image width
step=512
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

def get_loader(path):
    train_filename=os.listdir(path+"train")
    train=pd.DataFrame(columns=['speed', 'speed_med_5', 'speed_med_20','speed_SD_5','speed_SD_20',
                   'acceleration','acceleration_med_5', 'acceleration_med_20','acceleration_SD_5','acceleration_SD_20',
                   'angular_speed','angular_speed_med_5', 'angular_speed_med_20','angular_speed_SD_5','angular_speed_SD_20',
                   'angular_acceleration','angular_acceleration_med_5','angular_acceleration_med_20','angular_acceleration_SD_5','angular_acceleration_SD_20',
                   'angle_diff','angle_diff_med_5','angle_diff_med_20','angle_diff_SD_5','angle_diff_SD_20',
                   'tag'])
    for file in train_filename:
        data=pd.read_excel(path+"train/"+file)
        train=train.append(data)
    train_x=[]
    for i in range(len(train['speed'])):
        #temp_train =list(train.iloc[i,[0,1,2,3,4,5,6,7]])
        temp_train = list(train.iloc[i, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]])
        train_x.append(temp_train)
    if len(train_x)%step!=0:
        shortage=len(train_x)%step
        padding=step-shortage
        #print(padding)
        temp_train = [0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0,0]
        for j in range(padding):
            train_x.append(temp_train)
    train_x= [train_x[i:i+step] for i in range(0,len(train_x),step)]
    train_data = np.array(train_x)


    train_tag=[]
    for i in range(len(train['speed'])):
        temp_tag =list(train.iloc[i,[25]])
        train_tag+=temp_tag
    if len(train_tag)%step!=0:
        shortage=len(train_tag)%step
        padding=step-shortage
        #print(padding)
        temp_train = [0]
        for j in range(padding):
            train_tag+=temp_train
    train_tag= [train_tag[i:i+step] for i in range(0,len(train_tag),step)]
    train_tag = np.array(train_tag)
    val_filename=os.listdir(path+"val")
    val=pd.DataFrame(columns=['speed', 'speed_med_5', 'speed_med_20','speed_SD_5','speed_SD_20',
                   'acceleration','acceleration_med_5', 'acceleration_med_20','acceleration_SD_5','acceleration_SD_20',
                   'angular_speed','angular_speed_med_5', 'angular_speed_med_20','angular_speed_SD_5','angular_speed_SD_20',
                   'angular_acceleration','angular_acceleration_med_5','angular_acceleration_med_20','angular_acceleration_SD_5','angular_acceleration_SD_20',
                   'angle_diff','angle_diff_med_5','angle_diff_med_20','angle_diff_SD_5','angle_diff_SD_20',
                   'tag'])
    for file in val_filename:
        data=pd.read_excel(path+"val/"+file)
        val=val.append(data)
    dev=[]
    for i in range(len(val['speed'])):
        temp_dev =list(val.iloc[i,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]])
        dev.append(temp_dev)
    if len(dev)%step!=0:
        shortage=len(dev)%step
        padding=step-shortage
        #print(padding)
        temp_dev = [0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0,0]
        for j in range(padding):
            dev.append(temp_dev)
    dev= [dev[i:i+step] for i in range(0,len(dev),step)]
    #dev_data = torch.tensor(np.array(dev),dtype=torch.float32)
    dev_data=np.array(dev)
    #提取验证集标签
    dev_tag=[]
    for i in range(len(val['speed'])):
        temp_tag =list(val.iloc[i,[25]])
        dev_tag+=temp_tag
    if len(dev_tag)%step!=0:
        shortage=len(dev_tag)%step
        padding=step-shortage
        #print(padding)
        temp_dev = [0]
        for j in range(padding):
            dev_tag+=temp_dev
    dev_tag= [dev_tag[i:i+step] for i in range(0,len(dev_tag),step)]
    dev_tag = np.array(dev_tag)


    test_filename=os.listdir(path+"test")
    test=pd.DataFrame(columns=['speed', 'speed_med_5', 'speed_med_20','speed_SD_5','speed_SD_20',
                   'acceleration','acceleration_med_5', 'acceleration_med_20','acceleration_SD_5','acceleration_SD_20',
                   'angular_speed','angular_speed_med_5', 'angular_speed_med_20','angular_speed_SD_5','angular_speed_SD_20',
                   'angular_acceleration','angular_acceleration_med_5','angular_acceleration_med_20','angular_acceleration_SD_5','angular_acceleration_SD_20',
                   'angle_diff','angle_diff_med_5','angle_diff_med_20','angle_diff_SD_5','angle_diff_SD_20',
                   'tag'])
    for file in test_filename:
        data=pd.read_excel(path+"test/"+file)
        test=test.append(data)
    test_data=[]
    for i in range(len(test['speed'])):
        temp_test =list(test.iloc[i,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]])
        test_data.append(temp_test)
    if len(test_data)%step!=0:
        shortage=len(test_data)%step
        padding=step-shortage
        #print(padding)
        temp_test = [0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0,0]
        for j in range(padding):
            test_data.append(temp_test)
    test_data= [test_data[i:i+step] for i in range(0,len(test_data),step)]
    #test_data = torch.tensor(np.array(test_data),dtype=torch.float32)
    test_data=np.array(test_data)

    test_tag=[]
    for i in range(len(test['speed'])):
        temp_tag =list(test.iloc[i,[25]])
        test_tag+=temp_tag
    if len(test_tag)%step!=0:
        shortage=len(test_tag)%step
        padding=step-shortage
        #print(padding)
        temp_test = [0]
        for j in range(padding):
            test_tag+=temp_test
    test_tag= [test_tag[i:i+step] for i in range(0,len(test_tag),step)]
    test_tag = np.array(test_tag)

    train_data=torch.tensor(train_data,dtype=torch.float32)
    train_tag=torch.tensor(train_tag)
    dev_data=torch.tensor(dev_data,dtype=torch.float32)
    dev_tag=torch.tensor(dev_tag)
    test_data=torch.tensor(test_data,dtype=torch.float32)
    test_tag=torch.tensor(test_tag)

    train_dataset=Data.TensorDataset(train_data, train_tag)
    dev_dataset=Data.TensorDataset(dev_data, dev_tag)
    test_dataset=Data.TensorDataset(test_data, test_tag)

    train_loader = Data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    dev_loader = Data.DataLoader(dataset=dev_dataset,batch_size=batch_size,shuffle=True)
    test_loader = Data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)
    return train_loader,dev_loader,test_loader


DEVICE = "cuda:1"
model =LSTM(embedding_size=25, hidden_size=128, num_layers=1)
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay = WEIGHT_DECAY)
criteon = nn.CrossEntropyLoss()

final_indice=[]
final_y=[]

f_pred=[]
f_true=[]
def train(model,optimizer,train_loader):
    avg_loss = []
    model.train()
    for batch in train_loader:
        b_input_ids, b_labels = batch[0].cuda(), batch[1].cuda()           #shape均为[batch, intervals]
        #output = model(b_input_ids, b_labels)           #.forward_with_crf
        output = model(b_input_ids, b_labels)  # .forward_with_crf
        loss, logits = output[0], output[1]             #logits.shape = [batch, intervals, 2]
        avg_loss.append(loss.cpu().detach().numpy().item())
        #avg_loss.append(loss[0].cpu().detach().numpy().item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = np.array(avg_loss).mean()
    print('train_loss:',avg_loss)
def evaluate(model,loader,flag):
    # global final_indice,final_y
    model.eval()
    val_loss=[]
    # correct_all = []
    # true_label = []
    # pred_label = []
    with torch.no_grad():
        for batch in loader:
            b_input_ids, b_labels = batch[0].cuda(), batch[1].cuda()  # shape均为[batch, intervals]
            output = model(b_input_ids, b_labels)
            loss, logit = output[0], output[1]
            val_loss.append(loss.cpu().detach().numpy().item())
            _,logits = torch.max(logit, dim=2)
            # pred_label.append(logits.cpu().numpy())
            # true_label.append(b_labels.cpu().numpy())
            if flag=="test":
                f_pred.append(np.concatenate(logits.cpu().numpy()))
                f_true.append(np.concatenate(b_labels.cpu().numpy()))
                final_indice.append(np.concatenate(logits.cpu().numpy()))
                final_y.append(np.concatenate(b_labels.cpu().numpy()))
        # acc = accuracy_score(np.concatenate(true_label), np.concatenate(pred_label))
        loss=np.array(val_loss).mean()
        return loss
def train_begin(train_loader,dev_loader):
    max_val_f1=100000000000
    for epoch in range(NUM_EPOCHS):
        print("Epoch {}/{}".format(epoch + 1, NUM_EPOCHS))
        train(model, optimizer,train_loader)                        #, train_acc
        loss = evaluate(model, dev_loader,"val")
        print('val loss = ', loss)
        if loss < max_val_f1:
            max_val_f1 = loss
            torch.save(model.state_dict(), 'save_model/'+str(randomstate)+'paddy_bilstm.pth')
        print('beat loss = ', max_val_f1)
'''
def pre_true_ground():
    model.load_state_dict(torch.load('save_model/用来预测的model/best_acc_lstm_600_wheat_exchange_220.pth'))
    for file in test_filename:
        print(file)
        test = pd.DataFrame(columns=["lon", "lat", "dir", "vin", "speed", "ap", "jp", "br", "tag", "lon_diff", "lat_diff", "dir_diff"])
        data = pd.read_csv(path + file)
        test = test.append(data)
        test_data = []
        for i in range(len(test['lon'])):
            temp_test = list(test.iloc[i, [3, 4, 5, 6, 7, 9, 10, 11]])
            test_data.append(temp_test)
        if len(test_data) % step != 0:
            shortage = len(test_data) % step
            padding = step - shortage
            # print(padding)
            temp_test = [0, 0, 0, 0, 0, 0, 0, 0]
            for j in range(padding):
                test_data.append(temp_test)
        test_data = [test_data[i:i + step] for i in range(0, len(test_data), step)]
        test_data = np.array(test_data)
        test_tag = []
        for i in range(len(test['lon'])):
            temp_tag = list(test.iloc[i, [8]])
            test_tag += temp_tag
        if len(test_tag) % step != 0:
            shortage = len(test_tag) % step
            padding = step - shortage
            # print(padding)
            temp_test = [0]
            for j in range(padding):
                test_tag += temp_test
        test_tag = [test_tag[i:i + step] for i in range(0, len(test_tag), step)]
        test_tag = np.array(test_tag)
        test_data = torch.tensor(test_data, dtype=torch.float32)
        test_tag = torch.tensor(test_tag)
        test_dataset = Data.TensorDataset(test_data, test_tag)
        test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
        model.eval()
        val_loss = []
        correct_all = []
        final_indice = []
        final_y = []
        with torch.no_grad():
            for batch in test_loader:
                b_input_ids, b_labels = batch[0].cuda(), batch[1].cuda()  # shape均为[batch, intervals]
                output = model(b_input_ids, b_labels)
                loss, logit = output[0], output[1]
                val_loss.append(loss.cpu().detach().numpy().item())
                logits = torch.argmax(logit, dim=2)
                final_indice.append(np.concatenate(logits.cpu().numpy()))
                final_y.append(np.concatenate(b_labels.cpu().numpy()))
            val_loss = np.array(val_loss).mean()
            print("loss:",val_loss)
            y_final=np.concatenate(final_y)
            indice_final=np.concatenate(final_indice)
            print(classification_report(y_final, indice_final,digits=4))
            indice_final=list(indice_final)
            indice_excel=pd.DataFrame(indice_final)
            indice_excel.columns=['tag']
            #file = open(, "w")
            #indice_excel.to_csv('/home/liguangyuan/GCN_system/all_result/小麦/LSTM/'+file,index=False)
'''

if __name__=="__main__":
    kfold=10
    torch.manual_seed(0)
    accuracy_score_list=[]
    test_road_precision = 0
    test_road_recall = 0
    test_road_f1score = 0
    test_field_precision = 0
    test_field_recall = 0
    test_field_f1score = 0
    test_accuracy = 0
    test_macro_precision = 0
    test_macro_recall = 0
    test_macro_f1score = 0
    test_weight_precision = 0
    test_weight_recall = 0
    test_weight_f1score = 0
    lenset=kfold
    for randomstate in range(kfold):
        # randomstate=2021
        print(randomstate)
        path = '/xxx/xxx/xxx/xxx/xxx/xxx/xxx/paddy_100_25_10fold/'+str(randomstate)+"/"
        # spilt_data(path, randomstate)
        train_loader,dev_loader,test_loader=get_loader(path)
        train_begin(train_loader,dev_loader)
        model.load_state_dict(torch.load('save_model/'+str(randomstate)+'paddy_bilstm.pth'))
        evaluate(model,test_loader,"test")
        model.zero_grad()
        optimizer.zero_grad()
        model = LSTM(embedding_size=25, hidden_size=128, num_layers=1).cuda()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criteon = nn.CrossEntropyLoss()
        y_final=np.concatenate(final_y)
        indice_final=np.concatenate(final_indice)
        # print(classification_report(y_final, indice_final,digits=4,output_dict=True)['accuracy'])
        accuracy_score_list.append(classification_report(y_final, indice_final,digits=4,output_dict=True)['accuracy'])
        test_road_precision += classification_report(y_final, indice_final, digits=4, output_dict=True)['0']['precision']
        test_road_recall += classification_report(y_final, indice_final, digits=4, output_dict=True)['0']['recall']
        test_road_f1score += classification_report(y_final, indice_final, digits=4, output_dict=True)['0']['f1-score']
        test_field_precision += classification_report(y_final, indice_final, digits=4, output_dict=True)['1']['precision']
        test_field_recall += classification_report(y_final, indice_final, digits=4, output_dict=True)['1']['recall']
        test_field_f1score += classification_report(y_final, indice_final, digits=4, output_dict=True)['1']['f1-score']
        test_accuracy += classification_report(y_final, indice_final, digits=4, output_dict=True)['accuracy']
        test_macro_precision += classification_report(y_final, indice_final, digits=4, output_dict=True)['macro avg']['precision']
        test_macro_recall += classification_report(y_final, indice_final, digits=4, output_dict=True)['macro avg']['recall']
        test_macro_f1score += classification_report(y_final, indice_final, digits=4, output_dict=True)['macro avg']['f1-score']
        test_weight_precision += classification_report(y_final, indice_final, digits=4, output_dict=True)['weighted avg']['precision']
        test_weight_recall += classification_report(y_final, indice_final, digits=4, output_dict=True)['weighted avg']['recall']
        test_weight_f1score += classification_report(y_final, indice_final, digits=4, output_dict=True)['weighted avg']['f1-score']

        final_indice=[]
        final_y=[]
    print(accuracy_score_list)
    print('               precision    recall    f1score')
    print('        0    ' + str(round(test_road_precision / lenset, 4)) + '      ' + str(
        round(test_road_recall / lenset, 4)) + '    ' + str(round(test_road_f1score / lenset, 4)))
    print('        1    ' + str(round(test_field_precision / lenset, 4)) + '      ' + str(
        round(test_field_recall / lenset, 4)) + '    ' + str(round(test_field_f1score / lenset, 4)))
    print('\n')
    print('    accuracy                          ' + str(round(test_accuracy / lenset, 4)))
    print('   macro avg    ' + str(round(test_macro_precision / lenset, 4)) + '      ' + str(
        round(test_macro_recall / lenset, 4)) + '    ' + str(round(test_macro_f1score / lenset, 4)))
    print('weighted avg    ' + str(round(test_weight_precision / lenset, 4)) + '      ' + str(
        round(test_weight_recall / lenset, 4)) + '    ' + str(round(test_weight_f1score / lenset, 4)))
    #pre_true_ground()












