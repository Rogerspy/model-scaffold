#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utility.py
@Time    :   2022/05/24 22:22:33
@Author  :   Rogerspy
@Email   :   rogerspy@163.com
@Copyright : Rogerspy
'''


"""
Useful Python decorators for Data Scientists:
https://bytepawn.com/python-decorators-for-data-scientists.html?continueFlag=b37a171017f2251272527cb6fc2d9751
"""


import sys
from typing import *
from time import sleep, time
from random import seed
from io import StringIO
from sys import settrace
from random import randint
from math import log, floor
from itertools import chain
from functools import reduce
from socket import gethostname
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from contextlib import redirect_stdout
from timeit import default_timer as timer
from multiprocessing.pool import ThreadPool


import unicodedata


def parallel(
    func=None, 
    args=(), 
    merge_func=lambda x:x, 
    parallelism = cpu_count()
):
    """
    并行数据处理。

    Args:
        func (_type_, optional): _description_. Defaults to None.
        args (tuple, optional): _description_. Defaults to ().
        merge_func (_type_, optional): _description_. Defaults to lambdax:x.
        parallelism (_type_, optional): _description_. Defaults to cpu_count().
    """
    def decorator(func: Callable):
        def inner(*args, **kwargs):
            results = Parallel(n_jobs=parallelism)(
                delayed(func)(*args, **kwargs) for i in range(parallelism)
            )
            return merge_func(results)
        return inner
    if func is None:
        # decorator was used like @parallel(...)
        return decorator
    else:
        # decorator was used like @parallel, without parens
        return decorator(func)
    
    
def stacktrace(func=None, exclude_files=['anaconda']):
    """
    追踪函数运行过程。

    Args:
        func (_type_, optional): _description_. Defaults to None.
        exclude_files (list, optional): _description_. Defaults to ['anaconda'].
    
    Usage:
        ```
            def b():
                print('...')

            @stacktrace
            def a(arg):
                print(arg)
                b()
                return 'world'
        ```
        output:
            --> Executing: a('foo')
            foo
            --> Executing: b()
            ...
            --> Returning: b() -> None
            --> Returning: a('foo') -> 'world'

    """
    def tracer_func(frame, event, arg):
        co = frame.f_code
        func_name = co.co_name
        caller_filename = frame.f_back.f_code.co_filename
        if func_name == 'write':
            return # ignore write() calls from print statements
        for file in exclude_files:
            if file in caller_filename:
                return # ignore in ipython notebooks
        args = str(tuple([frame.f_locals[arg] for arg in frame.f_code.co_varnames]))
        if args.endswith(',)'):
            args = args[:-2] + ')'
        if event == 'call':
            print(f'--> Executing: {func_name}{args}')
            return tracer_func
        elif event == 'return':
            print(f'--> Returning: {func_name}{args} -> {repr(arg)}')
        return
    def decorator(func: Callable):
        def inner(*args, **kwargs):
            settrace(tracer_func)
            func(*args, **kwargs)
            settrace(None)
        return inner
    if func is None:
        # decorator was used like @stacktrace(...)
        return decorator
    else:
        # decorator was used like @stacktrace, without parens
        return decorator(func)
    
    
def traceclass(cls: type):
    """
    追踪类.

    Args:
        cls (type): _description_
        
    Usage:
        ```
            @traceclass
            class Foo:
                i: int = 0
                def __init__(self, i: int = 0):
                    self.i = i
                def increment(self):
                    self.i += 1
                def __str__(self):
                    return f'This is a {self.__class__.__name__} object with i = {self.i}'

            f1 = Foo()
            f2 = Foo(4)
            f1.increment()
            print(f1)
            print(f2)
        ```
        output:
            --> Executing: Foo::__init__()
            --> Executing: Foo::__init__()
            --> Executing: Foo::increment()
            --> Executing: Foo::__str__()
            This is a Foo object with i = 1
            --> Executing: Foo::__str__()
            This is a Foo object with i = 4
    """
    def make_traced(cls: type, method_name: str, method: Callable):
        def traced_method(*args, **kwargs):
            print(f'--> Executing: {cls.__name__}::{method_name}()')
            return method(*args, **kwargs)
        return traced_method
    for name in cls.__dict__.keys():
        if callable(getattr(cls, name)) and name != '__class__':
            setattr(cls, name, make_traced(cls, name, getattr(cls, name)))
    return cls


def is_whitespace(char):
    """Checks whether `char` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def is_control(char):
    """Checks whether `char` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def is_punctuation(char):
    """Checks whether `char` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def is_end_of_word(text):
    """Checks whether the last character in text is one of a punctuation, control or whitespace character."""
    last_char = text[-1]
    return bool(is_control(last_char) | is_punctuation(last_char) | is_whitespace(last_char))


def is_start_of_word(text):
    """Checks whether the first character in text is one of a punctuation, control or whitespace character."""
    first_char = text[0]
    return bool(is_control(first_char) | is_punctuation(first_char) | is_whitespace(first_char))