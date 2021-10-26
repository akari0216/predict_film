from sqlalchemy import create_engine
import pymysql
import pandas as pd
import numpy as np
from logger import logger
from model import df_predict
import datetime

today = datetime.date.today()

def to_sql(df,tablename):
    conn = create_engine("mysql+pymysql://root:123456@localhost/film_data?charset=utf8")
    df.to_sql(tablename,con = conn,if_exists = "append",index = False)

#计算预测票房占比
df_predict_table = pd.pivot_table(data = df_predict,index = ["cinema"],values = ["predict_bo"],agg_func = {"predict_bo":np.sum},fill_value = 0,margins = False)
df_predict_table = pd.DataFrame(df_predict_table.reset_index())
df_predict_table.rename(columns = {"predict_bo":"total_bo"},inplace = True)
df_predict = pd.merge(left = df_predict,right = df_predict_table,on = "cinema",how = "left")
df_predict["predict_bo_percent"] = np.round(df_predict["predict_bo"] / df_predict["total_bo"] * 100,2)
df_predict.drop(columns = ["total_bo"],axis = 1,inplace = True)

df_predict["fetch_date"] = str(today)
#写入数据库
to_sql(df_predict,"predict_film_cinema")
logger.info("predict data updated into database")