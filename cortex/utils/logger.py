#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   logger.py
@Time    :   2022/05/22 19:48:04
@Author  :   Rogerspy
@Email   :   rogerspy@163.com
@Copyright : Rogerspy
'''


import logging
import logging.handlers


class Logs(object):
    def __init__(self, log_file_path):
        # 获取模块名称，测试的时候直接控模块即可，
        # 但是在实际使用的情况下需要针对不同需要进行日志撰写的模块进行命名
        # 列如：通讯协议模块，测试模块，数据库模块，业务层模块，API调用模块
        # 可以考虑 __init__(self, model_name) 这样传入，然后再用一个list规定一下模块名称
        self.logger = logging.getLogger("")
        rotatingFileHandler = logging.handlers.RotatingFileHandler(
            filename=log_file_path,
            maxBytes=1024 * 1024 * 50,
            backupCount=5
        )
        # 设置输出格式
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s', 
            '%Y-%m-%d %H:%M:%S'
        )
        rotatingFileHandler.setFormatter(formatter)
        # 控制台句柄
        console = logging.StreamHandler()
        console.setLevel(logging.NOTSET)
        console.setFormatter(formatter)
        # 添加内容到日志句柄中
        self.logger.addHandler(rotatingFileHandler)
        self.logger.addHandler(console)
        self.logger.setLevel(logging.NOTSET)

    def info(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)
