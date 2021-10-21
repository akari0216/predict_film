import logging
import datetime

today = str(datetime.date.today())
log_path = "C:\\Users\\xieminchao\\Desktop\\dolby\\dolby_log_%s.txt" % today

def get_logger():
    #创建logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    #创建handler用于创建log文件
    #记得修改日志路径
    handler = logging.FileHandler(log_path)
    handler.setLevel(level = logging.INFO)
    #formatter用于设置日志格式
    formatter = logging.Formatter("%(asctime)s  %(name)s  %(levelname)s  %(message)s",datefmt = "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

logger = get_logger()