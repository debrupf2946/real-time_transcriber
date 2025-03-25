
from datetime import datetime

def get_current_time_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_current_time_string_with_milliseconds():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]