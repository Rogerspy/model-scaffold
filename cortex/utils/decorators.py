#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   decorators.py
@Time    :   2023/09/28 11:52:37
@Author  :   Rogerspy-CSong
@Version :   1.0
@Contact :   rogerspy@163.com
@License :   (C)Copyright 2023-2024, Rogerspy-CSong
'''

from typing import Callable
from sys import settrace
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from timeit import default_timer as timer
from multiprocessing.pool import ThreadPool


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