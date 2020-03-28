---
title: '[小问题系列]sql中引号转义的一些问题'
date: 2020-03-28 22:37:52
categories:
    - 小问题系列
tags: 
    - 数据库
---

背景： 用python将api获取的数据解析并入库
我觉得我需要更加优雅的解决方案。。

<!-- more -->

首先我有一个sql模版，模版里有的字段是字符串值。
这是问题来了
```python
sql = 'insert into atable where name = \"%s\"'
```
api取到的数据有两种，一种值正常的字符串，第二种是又加了一次转义（这种是我们理想的）
```python
name1 = '我叫\"王蔡勐\"'
name2 = '我叫\\"王蔡勐\\"'
print(name1,name2)
```

    我叫"王蔡勐" 我叫\"王蔡勐\"

第二种直接%可以就可以直接得到想要的sql：
然而name1:
```python
print(sql % name1)
```
    insert into atable where name = "我叫"王蔡勐""
进行处理得到了想要的结果。
```python
print(sql % name1.replace('\"','\\"'))
```

    insert into atable where name = "我叫\"王蔡勐\""

但是这时name2又不对了。
```python
print(sql % (name2.replace('\"','\\"')))
```

    insert into atable where name = "我叫\\"王蔡勐\\""

此时只好这样。。
name1 name2 都可以得到理想的sql
```python
print(sql % (name2.replace('\\"','\"').replace('\"','\\"')))
```

    insert into atable where name = "我叫\"王蔡勐\""



```python
print(sql % (name2.replace('\\"','\"').replace('\"','\\"')))
```

    insert into atable where name = "我叫\"王蔡勐\""




