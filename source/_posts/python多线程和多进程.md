---
title: python多线程和多进程
date: 2020-04-27 12:42:22
tags:
---


<!-- more -->

### 1. multithread vs. multiprocess

除了线程和进程的基本区别外，在python中，由于GIL的存在，multithread和multiprocess又一个很重要的区别就是multithread无法利用多核。 因此，我们需要根据任务的不同进行选择。

- multithread: io密集型任务， 比如多个线程从一个api拉去数据并存到数据库或文件。
- multiprocess: 计算密集型任务， 比如多个线程进行超参数搜索，需要大量计算，必须利用充分多核。

以上都是在实际工作中用得到场景。

### 2. 如何使用

可以通过继承Thread和Process累，通过run函数实现自己的功能。 也可以直接传入一个函数实例化。 下边的代码是用继承类再实例化的方式，两者用起来非常相似。
#### 2.1 multithread

```python
from threading import Thread
class MyThread(Thread):
    def __init__(self):
        super().__init__()
 
    def run(self):
        # start时运行
        pass
 
# 这里创建全局变量可以多线程共享，需要注意线程安全的问题
#提供了锁 from threading import Lock
if __name__ == '__main__':
    t1 = MyThread()
    t1.start()

    t1.join()# 等待线程执行完毕
```

#### 2.2 multiprocess

```python
from multiprocessing import Process
class MyProcess(Process):
    def __init__(self):
        super().__init__()  

 
    def run(self):  
        pass
        # start时运行
 
# 这里创建全局变量可以多进程共享，需要注意线程安全的问题
#提供了锁 from threading import Lock
if __name__ == '__main__':
    p = MyProcess()
    p.start()  
    p.join()
```

更多细节可以参考 <https://www.cnblogs.com/wuqiuming/p/9533567.html>