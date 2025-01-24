#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Copyright 2023 The International Digital Economy Academy (IDEA). CCNL team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@File    :   tracer.py
@Time    :   2024/11/13 14:01:50
@Author  :   songchao 
@Version :   1.0
@License :   (C)Copyright 2022-2024, CCNL-IDEA
@Contact :   songchao@idea.edu.cn
'''


from time import time
from sys import settrace
from functools import wraps
from typing import Callable

from cortex.utils import Logs


def function_logger(func=None, logger=None):
    """
    追踪函数运行过程。
    """
    if logger is None:
        logger = Logs()

    def decorator(func: Callable):

        def tracer_func(frame, event, arg):
            func_name = frame.f_code.co_name
            if event == 'call':
                logger.info(f'[Executing] Function `{func_name}` at Line: {frame.f_lineno}')
                return tracer_func
            elif event == 'return':
                logger.info(f'[Returned] Function `{func_name}` Result: {arg}')
            elif event == 'exception':
                logger.info(f'[Exception] Function `{func_name}` Error: {arg}')
            return arg

        @wraps(func)
        def inner(*args, **kwargs):
            settrace(tracer_func)
            res = func(*args, **kwargs)
            settrace(None)
            return res
        return inner

    if func is None:
        return decorator
    else:
        return decorator(func)
    
def class_logger(func=None, logger=None):
    """
    追踪类.
    """
    if logger is None:
        logger = Logs()

    def decorator(cls: type):

        def make_traced(cls: type, method_name: str, method: Callable):
            def traced_method(*args, **kwargs):
                logger.info(f'[Executing] Method `{cls.__name__}::{method_name}`')
                res = method(*args, **kwargs)
                logger.info(f'[Returned] Method `{cls.__name__}::{method_name}` Result: {res}')
                return res
            return traced_method

        for name in cls.__dict__.keys():
            if callable(getattr(cls, name)) and name != '__class__':
                setattr(cls, name, make_traced(cls, name, getattr(cls, name)))
        return cls
    # return decorator
    if func is None:
        return decorator
    else:
        return decorator(func)