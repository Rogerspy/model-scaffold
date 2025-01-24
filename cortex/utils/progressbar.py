#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   progressbar.py
@Time    :   2023/09/26 15:10:27
@Author  :   Rogerspy-CSong
@Version :   1.0
@Contact :   rogerspy@163.com
@License :   (C)Copyright 2023-2024, Rogerspy-CSong
'''

from typing import Optional, Any, Set
from rich.progress import (Progress , TaskID, 
    TextColumn, BarColumn, MofNCompleteColumn, TimeRemainingColumn, SpinnerColumn)


class progress(Progress):
    # Only one live display may be active at once
    _progress = Progress(
        SpinnerColumn(speed=2.),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("[progress.percentage]{task.percentage:>4.1f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn('ETA'),
        TimeRemainingColumn(),
        transient=True
    )

    def __init__(self, iterable, task_description = None):
        super().__init__()
        self._active_ids: Set[TaskID] = set()
        self.iterable = iterable
        self.task_description = task_description

    def __iter__(self):
        iterable = self.iterable
        task_description = self.task_description or "Processing"
        task = self.add_task(f"[blue]{task_description} ...", total=len(iterable))
        self.start()
        for obj in iterable:
            yield obj
            self.update(task, advance=1)
        self.update(task, visible=False)
        self.console.print(f"[green]{task_description} 已完成![/green]")
        self.stop()

    @classmethod
    def start(cls):
        cls._progress.start()

    @classmethod
    def stop(cls):
        cls._progress.stop()

    @property
    def tasks(self):
        return self._progress.tasks

    @staticmethod
    def _cat_description(description, max_length=33):
        mid = (max_length - 3) // 2
        return description if len(description) < max_length else f'{description[:mid]}...{description[-mid:]}'

    def add_task(
        self,
        description: str,
        start: bool = True,
        total: Optional[float] = None,
        completed: int = 0,
        visible: bool = True,
        **fields: Any,
    ) -> TaskID:
        task_id = self._progress.add_task(
            description=self._cat_description(description),
            start=start, 
            total=total, 
            completed=completed, 
            visible=visible, 
            **fields
        )
        self._active_ids.add(task_id)
        return task_id

    def update(
        self,
        task_id: TaskID,
        *,
        total: Optional[float] = None,
        completed: Optional[float] = None,
        advance: Optional[float] = None,
        description: Optional[str] = None,
        visible: Optional[bool] = None,
        refresh: bool = False,
        **fields: Any,
    ) -> None:
        if description:
            description = self._cat_description(description)
        self._progress.update(
            task_id, 
            total=total, 
            completed=completed, 
            advance=advance,
            description=description, 
            visible=visible, 
            refresh=refresh, 
            **fields
        )
        if self._progress.tasks[task_id].finished and task_id in self._active_ids:
            self._active_ids.remove(task_id)


if __name__ == "__main__":
    import time
    import random

    def process(chunk):
        time.sleep(0.1)

    a = [random.randint(1,20) for _ in range(5)]
    b = [random.randint(1,20) for _ in range(10)]
    idx = 0
    for x in progress(a, task_description='hello'):
        for chunk in progress(b, task_description='hello'):
            process(chunk)
        idx += 1
