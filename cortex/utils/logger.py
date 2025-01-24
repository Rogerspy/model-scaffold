#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   logger.py
@Time    :   2023/09/26 15:08:42
@Author  :   Rogerspy-CSong
@Version :   1.0
@Contact :   rogerspy@163.com
@License :   (C)Copyright 2023-2024, Rogerspy-CSong
'''


import logging
import logging.handlers


class Log(object):
    def __init__(self, log_file_path: str | None = None):
        # 获取模块名称，测试的时候直接控模块即可，
        # 但是在实际使用的情况下需要针对不同需要进行日志撰写的模块进行命名
        # 列如：通讯协议模块，测试模块，数据库模块，业务层模块，API调用模块
        # 可以考虑 __init__(self, model_name) 这样传入，然后再用一个list规定一下模块名称
        self.logger = logging.getLogger("")
        # 设置输出格式
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s', 
            '%Y-%m-%d %H:%M:%S'
        )
        # 控制台句柄
        console = logging.StreamHandler()
        console.setLevel(logging.NOTSET)
        console.setFormatter(formatter)
        self.logger.addHandler(console)
        # 日志文件
        if log_file_path:
            rotatingFileHandler = logging.handlers.RotatingFileHandler(
                filename=log_file_path,
                maxBytes=1024 * 1024 * 50,
                backupCount=5
            )
            rotatingFileHandler.setFormatter(formatter)
            # 添加内容到日志句柄中
            self.logger.addHandler(rotatingFileHandler)
        
        self.logger.setLevel(logging.NOTSET)

    def info(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)