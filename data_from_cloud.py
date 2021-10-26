from sshtunnel import SSHTunnelForwarder
import pymysql
from sqlalchemy import create_engine
import datetime
import pandas as pd

today = datetime.date.today()

def get_data_by_ssh():
    server = SSHTunnelForwarder(
        ssh_address_or_host=("1.12.243.7",22),
        ssh_username="root",
        ssh_password="JinYi2017*",
        remote_bind_address=("localhost",3306))
    server.start()

    conn = pymysql.connect(
        user="root",
        passwd="jy123456",
        host="localhost",
        db="film_data",
        port=server.local_bind_port)
    
    cursor = conn.cursor()
    sql = "select * from topcdb_film_info where fetch_date = '%s'" % str(today)
    res = pd.read_sql(sql,conn)
    df = pd.DataFrame(res)
    return df

def update_data(df):
    conn = create_engine("mysql+pymysql://root:jy123456@localhost/film_data?charset=utf8")
    df.to_sql("topcdb_film_info",con = conn,if_exists = "append",index = False)

if __name__ == "__main__":
    df_data = get_data_by_ssh()
    update_data(df_data)
    print("update topcdb film info data success")
    logger.info("update topcdb film from ssh info local data success")

