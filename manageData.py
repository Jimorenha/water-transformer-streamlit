import pandas as pd
import numpy as np
from numpy import array
import chinese_calendar
import datetime
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

'''获取用于训练的数据集的归一化尺度'''
# 划分数据集
def get_train_test(datapath, n_steps_in, n_steps_out):
    data = pd.read_csv(datapath)
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
            '''全部区域  96,21 --> 8,13'''
            seq_x, seq_y = data[i:end_ix, :], data[end_ix:out_end_ix, -13:]
            '''2_20-21年2个区域（长时间）用于部署的数据'''
            # 20年2月13至21年3月7为训练集
            if k <= 45506:
                train_x.append(seq_x)
                train_y.append(seq_y)
            # 21年3月7至21年5月7为验证集
            elif k > 45506 and k <= 45794:
                valid_x.append(seq_x)
                valid_y.append(seq_y)
            # 21年5月7至21年6月7为测试集
            else:
                test_x.append(seq_x)
                test_y.append(seq_y)
            k = k + 1
    train, valid, test = [train_x, train_y], [valid_x, valid_y], [test_x, test_y]
    return array(train_x), array(train_y), array(valid_x), array(valid_y), array(test_x), array(test_y)

# 数据归一化
def train_minmaxscaler(train, valid, test):
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
def train_get_data(datapath, n_steps_in, n_steps_out):
    train_x, train_y, valid_x, valid_y, test_x, test_y = get_train_test(datapath, n_steps_in, n_steps_out)
    # 特征数据x归一化
    # _X经过归一化后的三维特征数据，sc_X特征列的归一化属性
    train_X, valid_X, test_X, sc_X = train_minmaxscaler(train_x, valid_x, test_x)
    # 特征数据y归一化
    # _Y经过归一化后的三维特征数据，sc_Y特征列的归一化属性
    train_Y, valid_Y, test_Y, sc_Y = train_minmaxscaler(train_y, valid_y, test_y)
    scaler = [sc_X, sc_Y]
    return train_X, train_Y, valid_X, valid_Y, test_X, test_Y, sc_X, sc_Y

# 获取训练集的归一化尺度
@st.cache
def train_get_scaler(n_steps_in, n_steps_out):
    print('计算原训练模型的归一化尺度')
    datapath = 'datasets/20-21(13).csv'
    train_x, train_y, valid_x, valid_y, test_x, test_y, sc_x, sc_y = train_get_data(datapath, n_steps_in, n_steps_out)
    return sc_y

'''获取用于训练的数据集的归一化尺度'''


# 获取当前值之前的所有数据，并清除异常值
def get_area(area):
    # 获取全部水量数据，去除时间和重复时间点
    area_ = area.iloc[1:-1, 2:]
    # 生成水量数据列表
    count_day = len(area_.columns)
    count_time = 96 # 一天有96条
    waterlist = []
    for i in range(count_day):
        for j in range(count_time):
            item = area_.iloc[j, i]
            if item == 'END':
                break
            else:
                item_int = float(item)
                waterlist.append(item_int)
    # 生成dataframe
    water_df = pd.DataFrame(waterlist, columns=['area'])
    # 把小于0的异常值设为Nan
    water_df.area[water_df.area<0] = np.nan
    # 把Nan值先进行向后填充，再向前填充（ffill:向后填充，bfill:向前填充）
    water_df.fillna(method='ffill', inplace=True)
    water_df.fillna(method='bfill', inplace=True)
    return water_df

def get_df(datapath):
    YLEJ = pd.read_excel(datapath, sheet_name='1.悦来二级 悦来二级需水量')
    YLSJ_LS = pd.read_excel(datapath, sheet_name='2.悦来三级-鹿山 悦来三级-鹿山三级需水量')
    YLSJ_CY = pd.read_excel(datapath, sheet_name='3.悦来三级-翠云 悦来三级-翠云需水量')
    YLSJ = pd.read_excel(datapath, sheet_name='4.悦来四级 悦来四级需水量')
    YLLY_SJ = pd.read_excel(datapath, sheet_name='5.梁悦四级-人和 人和四级需水量')
    YLWJ = pd.read_excel(datapath, sheet_name='6.悦来五级 悦来五级需水量')
    LTEJ_LJYZ = pd.read_excel(datapath, sheet_name='7.梁沱二级 梁沱二级-兰家院子需水量')
    LTEJ_SSQ = pd.read_excel(datapath, sheet_name='8.梁沱二级 梁沱二级-松树桥需水量')
    LTSJ = pd.read_excel(datapath, sheet_name='9.梁沱三级 梁沱三级需水量')
    JBEJ = pd.read_excel(datapath, sheet_name='10.江北二级 江北二级需水量')
    JCSJ = pd.read_excel(datapath, sheet_name='11.江茶三级 江茶三级需水量')
    YBEJ = pd.read_excel(datapath, sheet_name='13.渝北二级 渝北二级需水量')
    YBSJ = pd.read_excel(datapath, sheet_name='14.渝北三级 渝北三级需水量')

    _YLEJ = get_area(YLEJ)
    _YLSJ_LS = get_area(YLSJ_LS)
    _YLSJ_CY = get_area(YLSJ_CY)
    _YLSJ = get_area(YLSJ)
    _YLLY_SJ = get_area(YLLY_SJ)
    _YLWJ = get_area(YLWJ)
    _LTEJ_LJYZ = get_area(LTEJ_LJYZ)
    _LTEJ_SSQ = get_area(LTEJ_SSQ)
    _LTSJ = get_area(LTSJ)
    _JBEJ = get_area(JBEJ)
    _JCSJ = get_area(JCSJ)
    _YBEJ = get_area(YBEJ)
    _YBSJ = get_area(YBSJ)

    # 把所有区域拼成一个df
    dataset_all = pd.concat([_YLEJ, _YLSJ_LS, _YLSJ_CY, _YLSJ, _YLLY_SJ, _YLWJ, _LTEJ_LJYZ, _LTEJ_SSQ, _LTSJ, _JBEJ, _JCSJ, _YBEJ, _YBSJ], axis=1)
    # 改列名
    dataset_all.columns = ['YLEJ', 'YLSJ_LS', 'YLSJ_CY', 'YLSJ', 'YLLY_SJ', 'YLWJ', 'LTEJ_LJYZ', 'LTEJ_SSQ', 'LTSJ',
                           'JBEJ', 'JCSJ', 'YBEJ', 'YBSJ']

    # # 把所有区域拼成一个新df
    # dataset_all = pd.concat([_YLSJ_LS, _YLSJ, _YLLY_SJ, _YLWJ, _LTEJ_LJYZ, _LTEJ_SSQ, _JBEJ, _YBEJ, _YBSJ, _YLEJ, _YLSJ_CY, _LTSJ, _JCSJ], axis=1)
    # # 改列名
    # dataset_all.columns = ['YLSJ_LS', 'YLSJ', 'YLLY_SJ', 'YLWJ', 'LTEJ_LJYZ', 'LTEJ_SSQ', 'JBEJ', 'YBEJ', 'YBSJ',
    #                        'YLEJ', 'YLSJ_CY', 'LTSJ', 'JCSJ']

    return dataset_all

def get_month_hour_day(date):
    month = []
    day = []
    hour = []
    weekday = []
    for i in range(len(date)):
        mon = date[i].month
        d = date[i].day
        h = date[i].hour
        week = date[i].weekday() + 1
        month.append(mon)
        day.append(d)
        hour.append(h)
        weekday.append(week)
    return month, hour, day, weekday

def get_holiday(date):
    holiday_list = []
    for item in date:
        #     day = datetime.datetime.strptime(item, "%Y/%m/%d")
        day = item
        demo_time = datetime.date(day.year, day.month, day.day)
        detail = chinese_calendar.get_holiday_detail(demo_time)
        if detail[0] == True:
            holiday_list.append('1')
        else:
            holiday_list.append('0')
    return holiday_list

def get_all_feature(date, dataset):
    month, hour, day, weekday = get_month_hour_day(date)
    holiday_list = get_holiday(date)
    dataset.insert(1, 'month', month)
    dataset.insert(2, 'day', day)
    dataset.insert(3, 'hour', hour)
    dataset.insert(4, 'weekday', weekday)
    dataset.insert(5, 'holiday', holiday_list)
    return dataset

def get_all_weather(weather_path):
    weather = pd.read_excel(weather_path)
    date = weather['date']
    month, hour, day, weekday = get_month_hour_day(date)
    weather.insert(1, 'month', month)
    weather.insert(2, 'day', day)
    weather.insert(3, 'hour', hour)
    return weather

# 1day—->2h
# 获取一天的数据
def get_2h_df(datapath, insert_time):
    dataset_all = get_df(datapath)
    dataset_2 = dataset_all.iloc[-insert_time:]
    dataset_2h = dataset_2.reset_index(drop=True)
    return dataset_2h

# 2day—->1day
# 获取两天的数据
def get_1day_df(datapath, insert_time):
    dataset_all = get_df(datapath)
    dataset_1 = dataset_all.iloc[-insert_time:]
    dataset_1day = dataset_1.reset_index(drop=True)
    return dataset_1day


