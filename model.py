import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader,Dataset

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from data_process import train_data,test_data,df_origin_film_data
from logger import logger

logger.info("model preparing")
#获取数据
def get_train_data(batch_size):
    df = train_data
    length = df.shape[0]
    if length % batch_size != 0:
        new_length = length - length % batch_size
        df = df.iloc[:new_length,:]

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    train_x = df.iloc[:,2:-1]
    train_y = df.iloc[:,-1]
    
    scaler_x.fit(train_x)
    scaler_y.fit(np.array(train_y).reshape(-1,1))
    train_x = pd.DataFrame(scaler_x.transform(train_x),columns=train_x.columns.tolist())
    train_y = scaler_y.transform(np.array(train_y).reshape(-1,1))

    return train_x,train_y,df.shape[0],scaler_x,scaler_y

# Hyper Parameters
EPOCH = 200  # 训练整批数据多少次
BATCH_SIZE = 50
LR = 0.001  # 学习率

#定义dataset
class trainDataset(Dataset):
    def __init__(self,data_x,data_y,length):
        self.data_x = np.array(data_x)
        self.data_y = np.array(data_y)
        self.length = length

    def __getitem__(self,idx):
        data_x = paddle.to_tensor(self.data_x[idx].reshape([1,20]),dtype="float32")
        data_y = paddle.to_tensor(self.data_y[idx],dtype="float32")
        return data_x,data_y

    def __len__(self):
        return self.length

train_x,train_y,length,scaler_x,scaler_y = get_train_data(BATCH_SIZE)
print(train_x.shape,train_y.shape)
train_data = trainDataset(train_x,train_y,length)


dataloader = DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)

class CNN(nn.Layer):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv1D(in_channels=1,out_channels=32,kernel_size=7,stride=2) #size[50,1,20] -> [50,32,8]
        self.conv2 = nn.Conv1D(in_channels=32,out_channels=16,kernel_size=7,stride=2) #size[50,32,8] -> [50,16,2]
        self.conv1T = nn.Conv1DTranspose(in_channels=16,out_channels=32,kernel_size=7,stride=2) #size[50,16,20] -> [50,32,8]
        self.conv2T = nn.Conv1DTranspose(in_channels=32,out_channels=1,kernel_size=7,stride=2) #size[50,32,8] -> [50,1,20] 19？？
        self.dnn = nn.Sequential(
            nn.Linear(1*19,32),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(32,1)
        )

    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.dropout(x,0.2)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv1T(x)
        x = F.relu(x)
        x = F.dropout(x,0.2)
        x = self.conv2T(x)
        x = paddle.reshape(x,shape=[x.shape[0],-1])
        output = self.dnn(x)
        return output

cnn = CNN()
print(cnn)

sheduler = paddle

mse_loss = nn.MSELoss()
optimizer = paddle.optimizer.Adam(learning_rate=LR,parameters=cnn.parameters())

for epoch in range(1,EPOCH + 1):
    print("=" * 50)
    print("epoch {}/{}\n".format(epoch,EPOCH))
    logger.info("model training...")
    epoch_loss = 0
    for step,(x,y) in enumerate(dataloader):
        output = cnn(x)
        loss = mse_loss(output,y)
        batch_loss = 0
        batch_loss += loss.numpy()[0]
        epoch_loss += batch_loss
        if step % 50 == 0:
            batch_loss /= BATCH_SIZE
            batch_loss *= 1000
            print("epoch:%d,step:%d,batch_loss:%.5f" % (epoch,step,batch_loss))
            batch_loss = 0

        optimizer.clear_grad()
        loss.backward()
        optimizer.step()
    
    epoch_loss /= length
    epoch_loss *= 1000
    print("epoch:%d,epoch_loss:%.4f" % (epoch,epoch_loss))


def get_test_data():
    df = test_data
    test_x = np.array(df.iloc[:,2:-1])
    test_y = np.array(df.iloc[:,-1])
    test_x = scaler_x.transform(test_x)

    return test_x,test_y,df.shape[0]

test_x,test_y,test_length = get_test_data()
test_data = trainDataset(test_x,test_y,test_length)
testDataLoader = DataLoader(dataset=test_data,batch_size=1,shuffle=False)

#预测
logger.info("model predicting")
predict = []
cnn.eval()
for step,(x,y) in enumerate(testDataLoader):
    output = cnn(x).numpy()
    predict.append(output[0])

predict = np.array(predict)
predict = scaler_y.inverse_transform(predict)
df_predict = df_origin_film_data.copy()
df_predict["predict_bo"] = predict