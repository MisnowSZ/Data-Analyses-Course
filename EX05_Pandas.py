import pandas as pd
from pandas import Series, DataFrame

data = {'Chinese':[66, 95, 95, 90, 80, 80], 'English':[65, 85, 92, 88, 90, 90], 'Math':[None, 98, 96, 77, 90, 90]}
#定义了index和columns对数据访问很方便
df_score = DataFrame(data, index=['ZhangFei', 'GuanYu', 'ZhaoYun', 'HuangZhong', 'DianWei', 'DianWei'], columns=['Chinese','English','Math'])

df_score = df_score.drop_duplicates()
df_score['Math'].fillna(df_score['Math'].mean(), inplace = True)
df_score['Sum'] = df_score['Chinese'] + df_score["Math"] + df_score["English"]
df_score = df_score.sort_values(by='Sum')
print(df_score)
