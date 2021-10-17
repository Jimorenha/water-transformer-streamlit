import streamlit as st
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split,TensorDataset
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import manageData
from tst import Transformer
from manageData import train_get_scaler

# Training parameters
BATCH_SIZE = 8
NUM_WORKERS = 0
LR = 2e-7
EPOCHS = 80

# Model parameters
q = 8  # Query size
v = 8  # Value size
h = 8  # Number of heads
N = 6  # Number of encoder and decoder to stack
attention_size = 96  # Attention window size
dropout = 0.2  # Dropout rate
pe ='original' # Positional encoding
chunk_mode = None

d_input = 21 # From dataset
d_model = 100  # Lattent dim
d_output = 13 # From dataset
n_steps_in = 96*2 # 输入时间维度
n_steps_out = 96 # 输出时间维度

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# 加载模型
@st.cache
def load_model(model_path):
    print('load model4')
    with torch.no_grad():
        style_model = Transformer(d_input, d_model, d_output, n_steps_out, q, v, h, N, attention_size=attention_size,
                  dropout=dropout, chunk_mode=chunk_mode, pe=pe).to(device)
        state_dict = torch.load(model_path, map_location='cpu')
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        style_model.eval()
        return style_model

def get_dataset(dataframe, n_steps_in, n_steps_out):
    data = dataframe
    # print(data)
    # 除去第一列时间列
    data = data.iloc[:, 1:]
    data = data.values
    dataset_x = []
    k = 1
    data_len = int(len(data))
    for i in range(data_len):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # a = np.isnan(data[i:out_end_ix, -1]).any()
        # if a:
        #     continue
        if end_ix <= len(data):
            seq_x = data[i:end_ix, :]
            dataset_x.append(seq_x)
            k = k + 1

    dataset = [dataset_x]
    return np.array(dataset_x)

def minmaxscaler(data):
    sc = MinMaxScaler()
    # sc_经过归一化后的二维数据
    sc_data = sc.fit_transform(data.reshape((-1, data.shape[-1])))
    # X转换成原来的三维数据
    datA = sc_data.reshape(data.shape)
    return datA, sc

def get_data(dataframe, n_steps_in, n_steps_out):
    dataset_x = get_dataset(dataframe, n_steps_in, n_steps_out)
    # 特征数据x归一化
    # _X经过归一化后的三维特征数据，sc_X特征列的归一化属性
    dataset_X, sc_X = minmaxscaler(dataset_x)
    # 特征数据y归一化
    # _Y经过归一化后的三维特征数据，sc_Y特征列的归一化属性
    return dataset_X, sc_X

# 反归一化
def reverse(data, scaler):
    if type(data) == torch.Tensor:
        data = data.numpy()
    if type(data) == torch.tensor:
        data = data.numpy()
    # item = [[[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0]]]
    # i = np.zeros([1, 96, 8])
    # temp = np.concatenate((i, data), axis=2)
    # temp = temp.reshape((-1, temp.shape[-1]))
    data = data.reshape((-1, data.shape[-1]))
    # 反转归一化
    y = scaler.inverse_transform(data)
    df = pd.DataFrame(y)
    # 返回最后13列作为水量预测值
    pre = df.iloc[:, -13:]
    # 修改模型预测出来的pre列名
    pre.columns = ['悦来三级-鹿山', '悦来四级', '梁悦四级-人和', '悦来五级',
                            '梁沱二级-兰家院子', '梁沱二级-松树桥', '江北二级', '渝北二级', '渝北三级', '悦来二级', '悦来三级-翠云', '梁沱三级', '江茶三级']
    return pre

def predict(model, dataframe, n_steps_in, n_steps_out, scaler):
    print('开始预测')
    dataset_x, sc_x_real = get_data(dataframe, n_steps_in, n_steps_out)
    # sc_y = train_get_scaler(96, 8, '2h')
    dataset = TensorDataset(torch.tensor(dataset_x))
    datasetloader = DataLoader(dataset,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True,
                                     num_workers=NUM_WORKERS,
                                     pin_memory=False
                                     )
    predictions = torch.empty(len(datasetloader.dataset), n_steps_out, d_output)
    idx_prediction = 0
    with torch.no_grad():
        for x in tqdm(datasetloader, total=len(datasetloader)):
            model = model.double()
            X = torch.cat(tuple(x), 0)
            netout = model(X.to(device)).cpu()
            predictions[idx_prediction:idx_prediction + X.shape[0]] = netout
            idx_prediction += X.shape[0]
    print('predictions的形状为：{}'.format(predictions.shape))
    print('----------------预测完毕-------------')
    pre = reverse(predictions, scaler)
    print('----------------反归一化完毕--------------')
    # 修改按文件sheet顺序读取的列顺序
    pre = pre[['悦来二级', '悦来三级-鹿山', '悦来三级-翠云', '悦来四级', '梁悦四级-人和', '悦来五级', '梁沱二级-兰家院子', '梁沱二级-松树桥', '梁沱三级', '江北二级', '江茶三级',
                           '渝北二级', '渝北三级']]
    return pre

# # 模型
# model = load_model('models/weather19_13(96x7_96).pth')
# # 天气
# weather_path = 'datasets/天气.xls'
# weather = manageData.get_all_weather(weather_path)
# # 需水量
# uploaded_file = 'datasets/报表查询.xls'
# dataset_7day = manageData.get_7day_df(uploaded_file)
# actual_date_7day = pd.date_range('2021/08/04 12:15', '2021/08/11 12:00', freq='15MIN')
# dataset_7day.insert(0, 'date', actual_date_7day)
# # 添加month/hour/holiday列
# data_7day = manageData.get_all_feature(actual_date_7day, dataset_7day)
# merge_7day = pd.merge(data_7day, weather, on=['month', 'hour', 'day'])
# result_7day = merge_7day[
#     ['date_x', 'month', 'hour', 'holiday', 'temp', 'humidity', 'rainfall', 'YLEJ', 'YLSJ_LS', 'YLSJ_CY', 'YLSJ',
#      'YLLY_SJ', 'YLWJ', 'LTEJ_LJYZ', 'LTEJ_SSQ', 'LTSJ',
#      'JBEJ', 'JCSJ', 'YBEJ', 'YBSJ']]
#
# pre = predict(model, result_7day, n_steps_in=96*7, n_steps_out=96)
