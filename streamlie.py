import streamlit as st
import pandas as pd
import numpy as np
import base64
import datetime as dt
from datetime import datetime
from datetime import timedelta
from datetime import timezone
# import datetime
import time
import manageData
import run_model3, run_model4

SHA_TZ = timezone(
    timedelta(hours=8),
    name='Asia/Shanghai',
)
# 协调世界时
utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
# 北京时间
beijing_now = utc_now.astimezone(SHA_TZ)
# print(beijing_now, beijing_now.tzname())

# 定义进入模型的预测时间
time_2h = 96
time_1day = 96*2

model_2h = "models/" + 'weather96_8' + ".pth"
model_1day = "models/" + 'weather21_13(96X2_96)' + ".pth"

print('----------------------------------------------------')
print('运行时间为:')
print(beijing_now, beijing_now.tzname())
print('----------------------------------------------------')
st.markdown('# 水务预测')
st.markdown('*水务区域预测，一共有两个模型，分别是：*')
st.markdown('*① 一天（96\*1条）预测未来2小时（8条）有天气信息的13个区域预测模型*')
st.markdown('*② 两天（96\*2条）预测未来一天（96条）有天气信息的13个区域预测模型*')
st.markdown('## 1. 准备数据')
st.markdown('### 1.1 请选择上传历史水量数据的最后日期时间点:')
st.markdown('***（注意：预测前请检查时间点，如果有错，请检查时间点）***')
# 日期
d = st.date_input(
    "日期：",
    )
# st.write('Your birthday is:', d)
# 时间
t = st.time_input('时间点：', dt.time(12, 00))
# st.write('Alarm is set for', t)
# 日期-时间
date_time = datetime.combine(d, t)
st.write('*历史水量数据的最后日期时间点为:*', date_time)

print('*******对进入模型的真实时间和预测时间进行整理********')
# 2day—->2h
actual_end_time_2h = date_time
# actual_end_time_2h
actual_start_time_2h = actual_end_time_2h - dt.timedelta(minutes=15 * (time_2h-1))
# actual_start_time_2h
actual_date_2h = pd.date_range(actual_start_time_2h, actual_end_time_2h, freq='15MIN')
print('预测未来两小时——真实数据开始时间为：{},真实数据结束时间为：{}，时间长度为：{}'.format(actual_start_time_2h, actual_end_time_2h, len(actual_date_2h)))

predict_start_time_2h = actual_end_time_2h + dt.timedelta(minutes=15)
# predict_start_time_2h
predict_end_time_2h = actual_end_time_2h + dt.timedelta(minutes=15 * 8)
# predict_end_time_2h
predict_date_2h = pd.date_range(predict_start_time_2h, predict_end_time_2h, freq='15MIN')
# predict_date_2h
print('预测未来两小时——预测数据开始时间为：{},预测数据结束时间为：{}，时间长度为：{}'.format(predict_start_time_2h, predict_end_time_2h, len(predict_date_2h)))

# 7day—->1day
actual_end_time_1day = date_time
# actual_end_time_1day
actual_start_time_1day = actual_end_time_1day - dt.timedelta(minutes=15 * (time_1day-1))
# actual_start_time_1day
actual_date_1day = pd.date_range(actual_start_time_1day, actual_end_time_1day, freq='15MIN')
# print(len(actual_date_1day))
print('预测未来一天——真实数据开始时间为：{},真实数据结束时间为：{}，时间长度为：{}'.format(actual_start_time_1day, actual_end_time_1day, len(actual_date_1day)))

predict_start_time_1day = actual_end_time_1day + dt.timedelta(minutes=15)
# predict_start_time_1day
predict_end_time_1day = actual_end_time_1day + dt.timedelta(minutes=15 * 96)
# predict_end_time_1day
predict_date_1day = pd.date_range(predict_start_time_1day, predict_end_time_1day, freq='15MIN')
# predict_date_1day
print('预测未来一天——预测数据开始时间为：{},预测数据结束时间为：{}，时间长度为：{}'.format(predict_start_time_1day, predict_end_time_1day, len(predict_date_1day)))
print('*******对进入模型的真实时间和预测时间进行整理完成********')

# 数据
st.markdown('### 1.2 请上传天气数据:')
st.markdown('***（注意：天气时间一小时一次，水量数据15分钟一次，请保持天气和水量数据时间(年/月/日 小时)一致,<font color=#f63366>将相应新天气数据填入模板</font>。例：天气时间点2021/08/11 12:00与水量时间点2021/08/11 12一致。）***', unsafe_allow_html=True)
# st.text_area('查询天气数据网站：http://hz.zc12369.com/home/meteorologicalData/dataDetails/ 将相应天气填入模板即可。')
st.markdown('查询天气数据网站：http://hz.zc12369.com/home/ ')

# expander = st.beta_expander("数据格式要求")
# expander.markdown('1.csv格式')
# expander.markdown('2.utf-8编码格式')
weather_path = st.file_uploader(label="请按照模板xls格式上传", type='xls', key='weather')
if weather_path is None:
    st.info('请上传文件')
if weather_path is not None:
    st.success('文件上传成功')
    # 所有数据的天气信息
    weather_all = manageData.get_all_weather(weather_path)
    # weather_all

st.markdown('### 1.3 请上传水量数据:')
st.markdown('***（注意：<font color=#f63366>将相应新水量数据填入模板</font>。请在水量数据每个分区最后日期时间点后加上<kbd>END</kbd>作为结束标志。'
         '例：若最后日期时间点为12:00，请在12:15单元格处输入<kbd>END</kbd>）***', unsafe_allow_html=True)
uploaded_file = st.file_uploader(label="请按照模板xls格式上传", type='xls', key='water')
if uploaded_file is None:
    st.info('请上传文件')
if uploaded_file is not None:
    st.success('文件上传成功')
    print('************************2day_2h数据整理开始*****************************')
    # 2day—->2h
    # 显示所有数据
    dataset_2h = manageData.get_2h_df(uploaded_file, time_2h)
    # 添加所有数据所对应的时间
    dataset_2h.insert(0, 'date', actual_date_2h)
    # 添加month/day/hour/weekday/holiday列
    data_2h = manageData.get_all_feature(actual_date_2h, dataset_2h)
    # 把两天数据和全部天气信息按照相同月、日、小时匹配
    merge_2h = pd.merge(data_2h, weather_all, on=['month', 'day', 'hour'])
    print('已与天气匹配')
    # 选取所需列
    result_2h = merge_2h[
        ['date_x', 'month', 'day', 'hour', 'weekday', 'holiday', 'temp', 'humidity', 'rainfall', 'YLEJ', 'YLSJ_LS', 'YLSJ_CY', 'YLSJ',
         'YLLY_SJ', 'YLWJ', 'LTEJ_LJYZ', 'LTEJ_SSQ', 'LTSJ',
         'JBEJ', 'JCSJ', 'YBEJ', 'YBSJ']]
    # 区域列重命名
    result_2h.columns = ['date', 'month', 'day', 'hour', 'weekday', 'holiday', 'temp', 'humidity', 'rainfall', '悦来二级', '悦来三级-鹿山',
                           '悦来三级-翠云', '悦来四级', '梁悦四级-人和', '悦来五级', '梁沱二级-兰家院子', '梁沱二级-松树桥', '梁沱三级', '江北二级', '江茶三级',
                           '渝北二级', '渝北三级']
    # 实际放入模型的预测df(model_2h)与按顺序展示的df不同
    model_2h_df = result_2h[['date', 'month', 'day', 'hour', 'weekday', 'holiday', 'temp', 'humidity', 'rainfall', '悦来三级-鹿山', '悦来四级', '梁悦四级-人和', '悦来五级',
                            '梁沱二级-兰家院子', '梁沱二级-松树桥', '江北二级', '渝北二级', '渝北三级', '悦来二级', '悦来三级-翠云', '梁沱三级', '江茶三级']]

    print('已获取最终所需列表')
    print('************************2day_2h数据整理结束*****************************')

    print('************************7day_1day数据整理开始*****************************')
    # 2day—->1day
    dataset_1day = manageData.get_1day_df(uploaded_file, time_1day)
    # 添加2天数据对应时间
    dataset_1day.insert(0, 'date', actual_date_1day)
    # 添加month/hour/holiday列
    data_1day = manageData.get_all_feature(actual_date_1day, dataset_1day)
    # 把2天数据和全部天气信息按照相同月、小时、日匹配
    merge_1day = pd.merge(data_1day, weather_all, on=['month', 'day', 'hour'])
    print('已与天气匹配')
    # 选取所需列
    result_1day = merge_1day[
        ['date_x', 'month', 'day', 'hour', 'weekday', 'holiday', 'temp', 'humidity', 'rainfall', 'YLEJ', 'YLSJ_LS', 'YLSJ_CY', 'YLSJ',
         'YLLY_SJ', 'YLWJ', 'LTEJ_LJYZ', 'LTEJ_SSQ', 'LTSJ',
         'JBEJ', 'JCSJ', 'YBEJ', 'YBSJ']]
    # 区域列重命名
    result_1day.columns = ['date', 'month', 'day', 'hour', 'weekday', 'holiday', 'temp', 'humidity', 'rainfall', '悦来二级', '悦来三级-鹿山',
                           '悦来三级-翠云', '悦来四级', '梁悦四级-人和', '悦来五级', '梁沱二级-兰家院子', '梁沱二级-松树桥', '梁沱三级', '江北二级', '江茶三级',
                           '渝北二级', '渝北三级']
    # 实际放入模型的预测df(model_2h)与按顺序展示的df不同
    model_1day_df = result_1day[['date', 'month', 'day', 'hour', 'weekday', 'holiday', 'temp', 'humidity', 'rainfall', '悦来三级-鹿山', '悦来四级', '梁悦四级-人和', '悦来五级',
                            '梁沱二级-兰家院子', '梁沱二级-松树桥', '江北二级', '渝北二级', '渝北三级', '悦来二级', '悦来三级-翠云', '梁沱三级', '江茶三级']]
    print('已获取最终所需列表')
    print('************************7day_1day数据整理结束*****************************')

    # 检查整理数据中是否有空值
    check_data = st.beta_expander("快捷检查整理数据中是否有空值")
    check_2h = result_2h[result_2h.isnull().T.any()]
    check_1day = result_1day[result_1day.isnull().T.any()]
    check_data.write(check_2h)
    check_data.write(check_1day)

    st.write(f'*真实一天数据时间段为：{actual_start_time_2h}到{actual_end_time_2h}*')
    water_2h = st.beta_expander("显示整理真实一天数据")
    water_2h.write(result_2h)
    # water_2day = st.beta_expander("显示模型预测真实一天数据")
    # water_2day.write(model_2h)

    st.write(f'*真实两天数据时间段为：{actual_start_time_1day}到{actual_end_time_1day}*')
    water_1day = st.beta_expander("显示整理真实两天数据")
    water_1day.write(result_1day)
    # water_7day = st.beta_expander("显示模型预测真实两天数据")
    # water_7day.write(model_1day)

st.markdown('## 2. 准备模型')
st.success('① 已选择两天（96*1条）预测未来2小时（8条）有天气信息的13个区域预测模型')
st.success('② 已选择七天（96*2条）预测未来一天（96条）有天气信息的13个区域预测模型')


# st.markdown('## 2. 准备模型')
# style_name = st.selectbox(
#     ' ',
#     ('无', 'weather19_13(96x2_8)', 'weather19_13(96x7_96)')
# )
# if style_name == '无':
#     st.info('请选择模型')
# elif style_name == 'weather19_13(96x2_8)':
#     st.success('① 已选择两天（96*2条）预测未来2小时（8条）有天气信息的13个区域预测模型')
# elif style_name == 'weather19_13(96x7_96)':
#     st.success('② 已选择七天（96*7条）预测未来一天（96条）有天气信息的13个区域预测模型')

# model= "models/" + style_name + ".pth"

st.markdown('## 3. 预测结果')
sc_x, sc_y = manageData.train_get_scaler(96, 8)
clicked = st.button('显示预测结果')
if weather_path is None:
    st.error('还未上传天气数据')
elif uploaded_file is None:
    st.error('还未上传水量数据')
else:
    if clicked:
        try:
            st.markdown('### 3.1 预测未来两小时')
            st.write(f'*预测未来两小时数据时间段为：{predict_start_time_2h}到{predict_end_time_2h}*')
            # 1day-->2h
            # 加载模型
            print('****************************2h模型开始***************************')
            model_predict2h = run_model3.load_model(model_2h)
            # 预测未来2h（8条）
            # 预测时间范围
            ori_df_2h = model_2h_df
            pre_2h = run_model3.predict(model_predict2h, model_2h_df, n_steps_in=time_2h, n_steps_out=8, sc_x=sc_x, sc_y=sc_y)
            # 添加时间列
            pre_2h.insert(0, 'date', predict_date_2h)
            st.markdown('#### 3.1.1 全部区域未来2小时（8条）预测需水量')
            pre_2h
            # 下载全部区域未来2小时（8条）预测需水量
            csv = pre_2h.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
            href = f'<a href="data:file/csv;base64,{b64}">下载全部区域未来2小时（8条）预测需水量</a> (右键链接另存为&lt;some_name&gt;.csv)'
            st.markdown(href, unsafe_allow_html=True)
            # 绘图
            plot_df_1 = pd.concat([ori_df_2h, pre_2h], axis=0)
            # 调整列顺序
            plot_df_1 = plot_df_1[['date', 'month', 'day', 'hour', 'weekday', 'holiday', 'temp', 'humidity', 'rainfall', '悦来二级', '悦来三级-鹿山',
                           '悦来三级-翠云', '悦来四级', '梁悦四级-人和', '悦来五级', '梁沱二级-兰家院子', '梁沱二级-松树桥', '梁沱三级', '江北二级', '江茶三级',
                           '渝北二级', '渝北三级']]
            # 丢弃绘图无关列
            plt_df_1 = plot_df_1.drop(['month', 'day', 'hour', 'weekday', 'holiday', 'temp', 'humidity', 'rainfall'], axis=1)
            # plt_df_1
            st.markdown('#### 3.1.2 真实1天+预测2小时全部区域水势绘制')
            wa = plt_df_1.iloc[:, 1:-1]
            st.line_chart(wa)
            plot_data_day_2h = st.beta_expander("真实1天+预测2小时全部区域水势绘制对应数据")
            plot_data_day_2h.write(plt_df_1)
            # 下载真实2天+预测2小时全部区域水势绘制对应数据
            csv = plt_df_1.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
            href = f'<a href="data:file/csv;base64,{b64}">下载真实1天+预测2小时全部区域水势绘制对应数据</a> (右键链接另存为&lt;some_name&gt;.csv)'
            st.markdown(href, unsafe_allow_html=True)
            # if st.checkbox('显示真实2天+预测2小时全部区域水势绘制对应数据'):

            st.markdown('#### 3.1.3 真实1天+预测2小时单个区域水势绘制')
            area_water = st.beta_expander("单个区域水势绘制")
            area_water.markdown('### 悦来二级')
            YLEJ = plt_df_1.iloc[:, 1]
            area_water.line_chart(YLEJ)
            area_water.markdown('### 悦来三级-鹿山')
            YLSJ_LS = plt_df_1.iloc[:, 2]
            area_water.line_chart(YLSJ_LS)
            area_water.markdown('### 悦来三级-翠云')
            YLSJ_CY = plt_df_1.iloc[:, 3]
            area_water.line_chart(YLSJ_CY)
            area_water.markdown('### 悦来四级')
            YLSJ = plt_df_1.iloc[:, 4]
            area_water.line_chart(YLSJ)
            area_water.markdown('### 梁悦四级-人和')
            LYSJ_RH = plt_df_1.iloc[:, 5]
            area_water.line_chart(LYSJ_RH)
            area_water.markdown('### 悦来五级')
            YLWJ = plt_df_1.iloc[:, 6]
            area_water.line_chart(YLWJ)
            area_water.markdown('### 梁沱二级-兰家院子')
            LTEJ_LJYZ = plt_df_1.iloc[:, 7]
            area_water.line_chart(LTEJ_LJYZ)
            area_water.markdown('### 梁沱二级-松树桥')
            LTEJ_SSQ = plt_df_1.iloc[:, 8]
            area_water.line_chart(LTEJ_SSQ)
            area_water.markdown('### 梁沱三级')
            LTSJ = plt_df_1.iloc[:, 9]
            area_water.line_chart(LTSJ)
            area_water.markdown('### 江北二级')
            JBEJ = plt_df_1.iloc[:, 10]
            area_water.line_chart(JBEJ)
            area_water.markdown('### 江茶三级')
            JCSJ = plt_df_1.iloc[:, 11]
            area_water.line_chart(JCSJ)
            area_water.markdown('### 渝北二级')
            YBEJ = plt_df_1.iloc[:, 12]
            area_water.line_chart(YBEJ)
            area_water.markdown('### 渝北三级')
            YBSJ = plt_df_1.iloc[:, 13]
            area_water.line_chart(YBSJ)
            print('****************************2h模型完毕***************************')


            # 7day-->1day
            st.markdown('### 3.2 预测未来一天')
            st.write(f'*预测未来一天数据时间段为：{predict_start_time_1day}到{predict_end_time_1day}*')
            # 加载模型
            print('****************************1day模型开始***************************')
            model_predict1day = run_model4.load_model(model_1day)
            # 预测未来2h（8条）
            # 预测时间范围
            ori_df_1day = model_1day_df
            pre_1day = run_model4.predict(model_predict1day, model_1day_df, n_steps_in=time_1day, n_steps_out=96, sc_x=sc_x, sc_y=sc_y)
            # print(pre_1day)
            # 添加时间列
            pre_1day.insert(0, 'date', predict_date_1day)
            st.markdown('#### 3.2.1 全部区域未来一天（96条）预测需水量')
            pre_1day
            # 下载全部区域未来一天（96条）预测需水量
            csv = pre_1day.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
            href = f'<a href="data:file/csv;base64,{b64}">下载全部区域未来一天（96条）预测需水量</a> (右键链接另存为&lt;some_name&gt;.csv)'
            st.markdown(href, unsafe_allow_html=True)
            # 绘图
            plot_df_7 = pd.concat([ori_df_1day, pre_1day], axis=0)
            # 调整列顺序
            plot_df_7 = plot_df_7[['date', 'month', 'day', 'hour', 'weekday', 'holiday', 'temp', 'humidity', 'rainfall', '悦来二级', '悦来三级-鹿山',
                           '悦来三级-翠云', '悦来四级', '梁悦四级-人和', '悦来五级', '梁沱二级-兰家院子', '梁沱二级-松树桥', '梁沱三级', '江北二级', '江茶三级',
                           '渝北二级', '渝北三级']]
            plt_df_7 = plot_df_7.drop(['month', 'day', 'hour', 'weekday', 'holiday', 'temp', 'humidity', 'rainfall'], axis=1)
            # plt_df_1
            st.markdown('#### 3.2.2 真实2天+预测一天全部区域水势绘制')
            wa_7_1 = plt_df_7.iloc[:, 1:-1]
            st.line_chart(wa_7_1)
            plot_data_7day_1day = st.beta_expander("真实2天+预测一天全部区域水势绘制对应数据")
            plot_data_7day_1day.write(plt_df_7)
            # 下载真实7天+预测一天全部区域水势绘制对应数据
            csv = plt_df_7.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
            href = f'<a href="data:file/csv;base64,{b64}">下载真实2天+预测一天全部区域水势绘制对应数据</a> (右键链接另存为&lt;some_name&gt;.csv)'
            st.markdown(href, unsafe_allow_html=True)

            # if st.checkbox('显示真实2天+预测2小时全部区域水势绘制对应数据'):

            st.markdown('#### 3.2.3 真实2天+预测一天单个区域水势绘制')
            area_water_7 = st.beta_expander("单个区域水势绘制")
            area_water_7.markdown('### 悦来二级')
            YLEJ = plt_df_7.iloc[:, 1]
            area_water_7.line_chart(YLEJ)
            area_water_7.markdown('### 悦来三级-鹿山')
            YLSJ_LS = plt_df_7.iloc[:, 2]
            area_water_7.line_chart(YLSJ_LS)
            area_water_7.markdown('### 悦来三级-翠云')
            YLSJ_CY = plt_df_7.iloc[:, 3]
            area_water_7.line_chart(YLSJ_CY)
            area_water_7.markdown('### 悦来四级')
            YLSJ = plt_df_7.iloc[:, 4]
            area_water_7.line_chart(YLSJ)
            area_water_7.markdown('### 梁悦四级-人和')
            LYSJ_RH = plt_df_7.iloc[:, 5]
            area_water_7.line_chart(LYSJ_RH)
            area_water_7.markdown('### 悦来五级')
            YLWJ = plt_df_7.iloc[:, 6]
            area_water_7.line_chart(YLWJ)
            area_water_7.markdown('### 梁沱二级-兰家院子')
            LTEJ_LJYZ = plt_df_7.iloc[:, 7]
            area_water_7.line_chart(LTEJ_LJYZ)
            area_water_7.markdown('### 梁沱二级-松树桥')
            LTEJ_SSQ = plt_df_7.iloc[:, 8]
            area_water_7.line_chart(LTEJ_SSQ)
            area_water_7.markdown('### 梁沱三级')
            LTSJ = plt_df_7.iloc[:, 9]
            area_water_7.line_chart(LTSJ)
            area_water_7.markdown('### 江北二级')
            JBEJ = plt_df_7.iloc[:, 10]
            area_water_7.line_chart(JBEJ)
            area_water_7.markdown('### 江茶三级')
            JCSJ = plt_df_7.iloc[:, 11]
            area_water_7.line_chart(JCSJ)
            area_water_7.markdown('### 渝北二级')
            YBEJ = plt_df_7.iloc[:, 12]
            area_water_7.line_chart(YBEJ)
            area_water_7.markdown('### 渝北三级')
            YBSJ = plt_df_7.iloc[:, 13]
            area_water_7.line_chart(YBSJ)
            print('****************************1day模型完毕***************************')
        except ValueError:
            st.error('请检查数据历史水量数据的最后日期时间点！！！')
