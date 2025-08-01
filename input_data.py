import numpy as np
import pandas as pd
import pickle as pkl

def load_sz_data(dataset):
    sz_adj = pd.read_csv(r'E:\Data\data/sz_adj.csv',header=None)
    adj = np.mat(sz_adj)
    sz_tf = pd.read_csv(r'E:\Data\data/sz_speed.csv')
    return sz_tf, adj

def load_los_data(dataset):
    los_adj = pd.read_csv(r'E:\I-24motiondata/flow.csv',header=None)
    adj = np.mat(los_adj)
    los_tf = pd.read_csv(r'E:\I-24motiondata/adj.csv')
    return los_tf, adj


def preprocess_data(data, time_len, rate, seq_len, pre_len):
    train_size = int(time_len * rate)
    train_data = data[0:train_size]
    test_data = data[train_size:time_len]
    
    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0 : seq_len])
        trainY.append(a[seq_len : seq_len + pre_len])
    for i in range(len(test_data) - seq_len -pre_len):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0 : seq_len])
        testY.append(b[seq_len : seq_len + pre_len])
      
    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    testX1 = np.array(testX)
    testY1 = np.array(testY)
    return trainX1, trainY1, testX1, testY1


def preprocess_data2(data, time_len, rate, seq_len, pre_len):
    train_size = int(time_len * rate)
    train_data = data[0:train_size]
    test_data = data[train_size:time_len]

    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0: seq_len])
        trainY.append(a[seq_len: seq_len + pre_len])

    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    return trainX1, trainY1
