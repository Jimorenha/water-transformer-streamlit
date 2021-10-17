import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from numpy import array
import numpy as np
import datetime
from matplotlib import pyplot as plt
import torch
import csv

'''分别对特征和标签同时归一化'''
# 数据归一化
def minmaxscaler(train, valid, test):
    sc = MinMaxScaler()
    # sc_经过归一化后的二维数据
    sc_train = sc.fit_transform(train.reshape((-1, train.shape[-1])))
    sc_valid = sc.transform(valid.reshape((-1, valid.shape[-1])))
    sc_test = sc.transform(test.reshape((-1, test.shape[-1])))
    # X转换成原来的三维数据
    traiN = sc_train.reshape(train.shape)
    valiD = sc_valid.reshape(valid.shape)
    tesT = sc_test.reshape(test.shape)
    return traiN, valiD, tesT, sc

# 获取划分了数据集后归一化的数据
def get_data(datapath, n_steps_in, n_steps_out):
    train_x, train_y, valid_x, valid_y, test_x, test_y = get_train_test(datapath, n_steps_in, n_steps_out)
    # 特征数据x归一化
    # _X经过归一化后的三维特征数据，sc_X特征列的归一化属性
    train_X, valid_X, test_X, sc_X = minmaxscaler(train_x, valid_x, test_x)
    # 特征数据y归一化
    # _Y经过归一化后的三维特征数据，sc_Y特征列的归一化属性
    train_Y, valid_Y, test_Y, sc_Y = minmaxscaler(train_y, valid_y, test_y)
    scaler = [sc_X, sc_Y]
    return train_X, train_Y, valid_X, valid_Y, test_X, test_Y, sc_X, sc_Y

# 划分数据集
def get_train_test(datapath, n_steps_in, n_steps_out):
    # datapath = 'd:/water.csv'
    data = read_csv(datapath)
    # 除去第一列时间列
    data = data.iloc[:, 1:]
    data = data.values
    train_x, train_y, valid_x, valid_y, test_x, test_y = [], [], [], [], [], []
    k = 1
    data_len = int(len(data))
    for i in range(data_len):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        a=np.isnan(data[i:out_end_ix, -1]).any()
        if a:
            continue
        if out_end_ix < len(data):
            '''全部区域  96,15/19 --> 96,9/13'''
            seq_x, seq_y = data[i:end_ix, :], data[end_ix:out_end_ix, 6:]
            '''只有水量  96,9 --> 96,9'''
            # seq_x, seq_y = data[i:end_ix, :], data[end_ix:out_end_ix, :]
            '''一个区域  96,7 --> 96,1'''
            # seq_x, seq_y = data[i:end_ix, :], data[end_ix:out_end_ix, -1]
            # seq_y = seq_y.reshape((-1, 1))
            '''添加过去天气和水量'''
            # seq_x, seq_y = data[i:end_ix, :], data[end_ix:out_end_ix, -9:]
            '''添加过去未来天气和过去水量'''
            # seq_x, seq_y = data[i:end_ix, :], data[end_ix:out_end_ix, -9:]
            '''9个区域'''
            if k <= 35137:
                train_x.append(seq_x)
                train_y.append(seq_y)
            elif k > 35137 and k <= 43777:
                valid_x.append(seq_x)
                valid_y.append(seq_y)
            else:
                test_x.append(seq_x)
                test_y.append(seq_y)
            # '''13个区域'''
            # if k <= 17569:
            #     train_x.append(seq_x)
            #     train_y.append(seq_y)
            # elif k > 17569 and k <= 20545:
            #     valid_x.append(seq_x)
            #     valid_y.append(seq_y)
            # else:
            #     test_x.append(seq_x)
            #     test_y.append(seq_y)
            k = k + 1
    train, valid, test = [train_x, train_y], [valid_x, valid_y], [test_x, test_y]
    return array(train_x), array(train_y), array(valid_x), array(valid_y), array(test_x), array(test_y)

# 选取每隔序列的第一个存入csv
def pick_one(data, result_path, filename, scaler):
    if type(data) == torch.Tensor:
        data = data.numpy()
    if type(data) == torch.tensor:
        data = data.numpy()
    # data = data.numpy()
    # 取每隔序列的第一个
    Y = []
    for i in range(len(data)):
        y = data[i][0]
        Y.append(y)
        # print(data[i][0])
    # Y = np.array(Y)
    # 反转归一化
    y = scaler.inverse_transform(Y)
    # 转化成DataFrame存入csv
    file_path = f'{result_path}/{filename}_{datetime.datetime.now().strftime("%Y_%m_%d__%H%M%S")}.csv'
    df = pd.DataFrame(y)
    # 取消索引，保留一位小数
    df.to_csv(file_path, index=False, float_format='%.1f')
    # 转化成ndarray返回最后一列
    npa = np.array(df)
    # 返回反归一化后的最后一列水势数据用于绘图
    return npa[:, -1]

def pick_all(data, result_path, filename, scaler):
    if type(data) == torch.Tensor:
        data = data.numpy()
    if type(data) == torch.tensor:
        data = data.numpy()
    data = data.reshape((-1, data.shape[-1]))
    # Y = np.array(data)
    # 反转归一化
    y = scaler.inverse_transform(data)
    # 转化成DataFrame存入csv
    file_path = f'{result_path}/{filename}_{datetime.datetime.now().strftime("%Y_%m_%d__%H%M%S")}.csv'
    # df = pd.DataFrame(y)
    # 取消索引，保留一位小数
    # df.to_csv(file_path, index=False, float_format='%.1f')
    # 转化成ndarray返回最后一列
    # npa = np.array(df)
    # 返回反归一化后的最后一列水势数据用于绘图
    water = pd.DataFrame(y)
    water.to_csv(file_path, index=False, float_format='%.1f')
    water_array = np.array(water)
    return water_array

# 绘制真实-预测对比曲线图
def comp_curve(actual, result_path, predictions, length, filename):
    actual = np.array(actual)
    predictions = np.array(predictions)
    fig = plt.figure(figsize=(20, 6))
    plt.plot(actual[:length], 'o-', label='actual')
    plt.plot(predictions[:length], 'o-', label='prediction')
    plt.legend()
    name = f'{result_path}/{filename}_{datetime.datetime.now().strftime("%Y_%m_%d__%H%M%S")}.png'
    plt.savefig(name)
    plt.show()
