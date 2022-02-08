# !/user/bin/env python
# coding=utf-8

# 1. pip install minio -i https://pypi.douban.com/simple
# 2. pip install oss2 -i https://pypi.douban.com/simple

import traceback
from minio import Minio
from minio.error import S3Error
import oss2


def minio_file_upload(end_point: str, access_key: str, secret_key: str,
                      bucket: str, remote_path: str, local_path: str):
    try:
        _end_point = end_point.replace('https://', '').replace('http://', '')
        # Create a client with the MinIO server playground, its access key
        # and secret key.
        client = Minio(_end_point,
                       access_key=access_key,
                       secret_key=secret_key,
                       secure=False)

        # Make 'asiatrip' bucket if not exist.
        found = client.bucket_exists(bucket)
        if not found:
            client.make_bucket(bucket)
        else:
            print("Bucket {} already exists".format(bucket))

        # Upload '/home/user/Photos/asiaphotos.zip' as object name
        # 'asiaphotos-2015.zip' to bucket 'asiatrip'.
        client.fput_object(
            bucket,
            remote_path,
            local_path,
        )
        print("{} is successfully uploaded as "
              "object {} to bucket {}.".format(local_path, remote_path,
                                               bucket))
    except Exception as e:
        print("*** minio上传文件异常 -> {} {}".format(str(e),
                                                traceback.format_exc()))
        raise Exception("minio上传文件异常:[{}]".format(str(e)))


def oss_file_upload(end_point: str, access_key: str, secret_key: str,
                    bucket: str, remote_path: str, local_path: str):
    try:
        _end_point = end_point.replace('https://', '').replace('http://', '')
        # 阿里云账号AccessKey拥有所有API的访问权限，风险很高。强烈建议您创建并使用RAM用户进行API访问或日常运维，请登录RAM控制台创建RAM用户。
        auth = oss2.Auth(access_key, secret_key)
        # yourEndpoint填写Bucket所在地域对应的Endpoint。以华东1（杭州）为例，Endpoint填写为https://oss-cn-hangzhou.aliyuncs.com。
        # 填写Bucket名称。
        bucket = oss2.Bucket(auth, _end_point, bucket)

        # 填写Object完整路径和本地文件的完整路径。Object完整路径中不能包含Bucket名称。
        # 如果未指定本地路径，则默认从示例程序所属项目对应本地路径中上传文件。
        bucket.put_object_from_file(remote_path, local_path)
        
    except Exception as e:
        print("*** oss上传文件异常 -> {} {}".format(str(e), traceback.format_exc()))
        raise Exception("oss上传文件异常:[{}]".format(str(e)))
    
def oss_folder_upload(end_point: str, access_key: str, secret_key: str,
                    bucket: str, remote_path: str, local_path: str):
    try:
        _end_point = end_point.replace('https://', '').replace('http://', '')
        # 阿里云账号AccessKey拥有所有API的访问权限，风险很高。强烈建议您创建并使用RAM用户进行API访问或日常运维，请登录RAM控制台创建RAM用户。
        auth = oss2.Auth(access_key, secret_key)
        # yourEndpoint填写Bucket所在地域对应的Endpoint。以华东1（杭州）为例，Endpoint填写为https://oss-cn-hangzhou.aliyuncs.com。
        # 填写Bucket名称。
        bucket = oss2.Bucket(auth, _end_point, bucket)

        # 填写Object完整路径和本地文件的完整路径。Object完整路径中不能包含Bucket名称。
        # 如果未指定本地路径，则默认从示例程序所属项目对应本地路径中上传文件。
        bucket.put_object_from_file(remote_path, local_path)
        
    except Exception as e:
        print("*** oss上传文件异常 -> {} {}".format(str(e), traceback.format_exc()))
        raise Exception("oss上传文件异常:[{}]".format(str(e)))