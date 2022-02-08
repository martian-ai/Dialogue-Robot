from utils import oss_file_upload
from config_oss import end_point, access_key, secret_key, bucket_name


remote_path = "test.txt"
local_path = "test.txt"
# oss_file_upload(end_point=end_point, access_key=access_key, secret_key=secret_key, bucket=bucket, remote_path=remote_path, local_path=local_path)

oss_file_upload(end_point=end_point, access_key=access_key, secret_key=secret_key, bucket=bucket_name, remote_path=remote_path, local_path=local_path)

# 设置为http 才行
# https://oss.console.aliyun.com/bucket/oss-cn-hangzhou/bot-mvp-resources/permission/policy