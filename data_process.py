import pandas as pd
import os
import re
import pymysql
import datetime
from ftplib import FTP
from logger import logger

import warnings
warnings.filterwarnings("ignore")

data_path = ""

#当预测周期为2天的时候再另行设定date_diff = 2

#获取ftp数据
def get_ftp_data(date):
    ftp = FTP()
    ftp.connect(host = "172.20.240.195",port = 21 ,timeout = 30)
    ftp.login(user = "sjzx",passwd = "jy123456@")
    ftp.set_pasv(False)
    list = ftp.nlst()
    date = date.replace("-","")
    filename = "SessionRevenue_"+ date +".csv"
    for each_file in list:
        judge = re.match(filename,each_file)
        if judge:
            file_handle = open(filename,"wb+")
            ftp.retrbinary("RETR "+filename,file_handle.write)
            file_handle.close()

    ftp.quit()
    return filename

#读取sql数据
def get_sql_data(sql):
    conn = pymysql.connect(host = "localhost",port = 3306,user = "root",passwd = "jy123456",db = "film_data",charset = "utf8")
    df = pd.read_sql(sql,conn)
    return df

#影城影片预售数据
def get_presale_data(date,date_diff = 1):
    fetch_date = date - datetime.timedelta(days = date_diff)
    sql = "select cinema,film,bo,session,occupancy,avg_price,presale_date from presale_film_cinema where presale_date = '%s' and fetch_date = '%s'" % (date,fetch_date)
    df_presale = get_sql_data(sql)
    return df_presale

#影城影片最终票房数据
def get_target_data(date):
    sql = "select cinema,film,bo as 'target' from film_cinema where op_date = '%s'" % date
    df_target_data = get_sql_data(sql)
    df_target_data["film"] = df_target_data["film"].str.replace(":","").str.replace("：","").str.replace("·","").str.replace(".","")
    return df_target_data
    
#档期分类
def get_shedule_cls():
    sql = "select shedule_date,shedule_name,shedule_lv from shedule_date_list"
    df_shedule_list = get_sql_data(sql)
    return df_shedule_list

#影片信息
def get_film_info(date,date_diff = 1):
    #需要对历史影片当前票房做归一化，因为预售数据的当前数据值一般都远小于历史的当前票房值
    fetch_date = date - datetime.timedelta(days = date_diff)
    #当固定了每天获取到第二天及第三天的影片预售数据时，再加入fetch_date
    sql = "select film,maoyan_score,current_bo,avg_price as 'film_avg_price',people_per_session as 'film_people_per_session',main_type,country,film_length,show_days from topcdb_film_info \
        where film_date = '%s'" % date
    df_film_info = get_sql_data(sql)
    main_type_list = ["动作","科幻","剧情","喜剧","动画","奇幻","爱情","战争","惊悚","纪录","戏曲"]
    main_type_dict = {key:main_type_list.index(key) for key in main_type_list}
    country_list = ["中国大陆","美国","日本","印度","中国香港","英国","中国台湾","其他"]
    country_dict = {key:country_list.index(key) for key in country_list}
    df_film_info["main_type"] = df_film_info["main_type"].map(main_type_dict)
    df_film_info["country"] = df_film_info["country"].apply(lambda x:"其他" if x not in country_list[:-1] else x)
    df_film_info["country"] = df_film_info["country"].map(country_dict)
    return df_film_info
    
#影院信息
def get_cinema_info():
    sql = "select cinema_name,hall_count,imax_hall_count,unimax_hall_count,city_lv,open_date from jycinema_info"
    df_cinema_info = get_sql_data(sql)
    return df_cinema_info

#黄金场次
def get_golden_session(date):
    sql = ""
    df_shedule_list = get_shedule_cls()
    df_shedule_lv_first = df_shedule_list[df_shedule_list["shedule_lv"] == 1]
    if date in df_shedule_lv_first["shedule_date"].tolist():
        shedule_name = df_shedule_lv_first[df_shedule_lv_first["shedule_date"] == date]["shedul_name"].tolist()[0]
        if shedule_name == "国庆节":
            sql = "select cinema_name,golden_time from golden_session_list where is_nationalday = 1"
        elif shedule_name == "春节":
            sql = "select cinema_name,golden_time from golden_session_list where is_springfes = 1"
    else:
        weekday = date.weekday() + 1
        if weekday in [1,2,3,4]:
            sql = "select cinema_name,golden_time from golden_session_list where is_weekday = 1"
        elif weekday in [5,6,7]:
            sql = "select cinema_name,golden_time from golden_session_list where is_weekend = 1"
            
    df_golden_session = get_sql_data(sql)
    return df_golden_session

#获取csv数据，用以判定黄金场时间段
def get_csv_presale_data(date,date_diff = 1):
    fetch_date = date - datetime.timedelta(days = date_diff)
    filename = get_ftp_data(str(fetch_date))
    df = pd.read_csv(filename)
    pat = "（.*?）\s*|\(.*?\)\s*|\s*"
    df["影片"].replace(pat,"",regex = True,inplace = True)
    df["影片"] = df["影片"].str.replace(":","").str.replace("：","").str.replace("·","").str.replace(".","")
    df["场次日期"] = pd.to_datetime(df["场次时间"],format = "%Y-%m-%d").astype(str).str.slice(0,10)
    df["场次时间"].replace("\d\d\d\d-\d\d-\d\d ","",regex = True,inplace = True)
    df["场次时间"] = df["场次时间"].str.slice(0,5)
    #影院名转换
    sql = "select vista_cinema_name,cinema_name from jycinema_info where op_status = 1"
    df_cinema_name = get_sql_data(sql)
    cinema_name_dict = df_cinema_name.set_index("vista_cinema_name").T.to_dict("records")[0]
    df["影院"] = df["影院"].map(cinema_name_dict)
    #筛选
    df = df[df["场次日期"].isin([str(date)]) & df["场次状态"].isin(["开启"])]
    df = df[["影院","影片","场次日期","场次时间"]]
    return df

#数据处理过程
def process_data(start_date,end_date,is_test = False):
    #档期列表
    df_shedule_list = get_shedule_cls()
    #影城信息
    df_cinema_info = get_cinema_info()
    field_list = ["cinema","film","bo","session","occupancy","avg_price","shedule_lv","is_weekend","maoyan_score","current_bo","film_avg_price","film_people_per_session","main_type",\
                  "country","film_length","show_days","hall_count","imax_hall_count","unimax_hall_count","city_lv","opened_days","have_golden_session","target"]
    df_total = pd.DataFrame(columns = field_list)
    datelist = pd.date_range(start = start_date,end = end_date,freq = "D")
    df_origin_film_data = pd.DataFrame()
    #按天处理
    for each_date in datelist:
        print("processing the date:%s" % each_date)
        each_date = datetime.date(each_date.year,each_date.month,each_date.day)
        df_presale_data = get_presale_data(each_date)
        shedule_dict = df_shedule_list[["shedule_date","shedule_lv"]].set_index("shedule_date").T.to_dict("records")[0]
        df_presale_data["shedule_lv"] = df_presale_data["presale_date"].apply(lambda x:shedule_dict[x] if x in shedule_dict.keys() else 3)
        df_presale_data["is_weekend"] = df_presale_data["presale_date"].apply(lambda x:1 if each_date.weekday() + 1 in [5,6,7] else 0)
        #备份测试集的原影片
        if is_test:
            df_origin_film_data = df_presale_data[["cinema","film","presale_date"]]
            df_origin_film_data.rename(columns = {"presale_date":"predict_date"},inplace = True)
        #影片对齐部分
        df_presale_data["film"] = df_presale_data["film"].str.replace(":","").str.replace("：","").str.replace("·","").str.replace(".","")
        df_film_info = get_film_info(each_date)
        df_film_info["film"] = df_film_info["film"].str.replace(":","").str.replace("：","").str.replace("·","").str.replace(".","")
        df_presale_data = pd.merge(left = df_presale_data,right = df_film_info,on = "film",how = "left")
        df_presale_data["show_days"] = df_presale_data["show_days"].apply(lambda x:365 if x > 365 else x)
        
        df_presale_data = pd.merge(left = df_presale_data,right = df_cinema_info,left_on = "cinema",right_on = "cinema_name",how = "left")
        df_presale_data["opened_days"] = each_date - df_presale_data["open_date"]
        df_presale_data["opened_days"] = df_presale_data["opened_days"].apply(lambda x:x.days)
        #黄金场处理和判定
        df_csv_data = get_csv_presale_data(each_date)
        df_csv_data["匹配"] = df_csv_data["影院"] + df_csv_data["场次时间"].str.slice(0,2)
        df_golden_session = get_golden_session(each_date)
        df_golden_session["golden_time"] = df_golden_session["golden_time"].astype(str).str.slice(7,9)
        df_golden_session["match"] = df_golden_session["cinema_name"] + df_golden_session["golden_time"]
        golden_session_list = df_golden_session["match"].tolist()
        df_csv_data = df_csv_data[df_csv_data["匹配"].isin(golden_session_list)]
        df_golden_cinema_film = df_csv_data[["影院","影片"]]
        df_golden_cinema_film.drop_duplicates(keep="first", inplace = True)
        df_golden_cinema_film["影院影片"] = df_golden_cinema_film["影院"] + df_golden_cinema_film["影片"]
        golden_cinema_film_list = df_golden_cinema_film["影院影片"].tolist()
        
        df_presale_data["cinema_film"] = df_presale_data["cinema"] + df_presale_data["film"]
        df_presale_data["have_golden_session"] = df_presale_data["cinema_film"].apply(lambda x:1 if x in golden_cinema_film_list else 0)
        
        #目标票房值
        df_target_data = get_target_data(each_date)
        df_target_data["cinema_film"] = df_target_data["cinema"] + df_target_data["film"]
        df_target_data = df_target_data[["cinema_film","target"]]
        df_presale_data = pd.merge(left = df_presale_data,right = df_target_data,on = "cinema_film",how = "left")
        df_presale_data.drop(columns = ["cinema_name","open_date","cinema_film","presale_date"],axis = 1,inplace = True)
        #去除匹不上的
        df_presale_data = df_presale_data[df_presale_data["current_bo"].notnull()]
        df_presale_data = df_presale_data[df_presale_data["target"].notnull()]
        #合并
        df_total = pd.concat([df_total,df_presale_data],ignore_index = True)

    if is_test:
        return df_total,df_origin_film_data
    else: 
        return df_total

today = datetime.date.today()
#用今天之前的10天数据作为训练数据
print("preparing train data")
start_date = str(today - datetime.timedelta(days = 11)).replace("-","")
end_date = str(today - datetime.timedelta(days = 1)).replace("-","")
train_data = process_data(start_date,end_date)
logger.info("train data prepared")

#预测用数据
print("preparing predict data")
test_date = str(today + datetime.timedelta(days = 1)).replace("-","")
test_data,df_origin_film_data = process_data(test_date,test_date,is_test=True)
logger.info("test data prepared")
