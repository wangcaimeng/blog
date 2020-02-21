---
title: Tensorboard远程访问
date: 2020-02-20 16:01:54
categories: 环境搭建
---
`tensorboard --logdir logs`默认启动是localhost，那么如何远程访问呢，网上搜到的都是配置ssh隧道。。然而其实官方提供了方案。

<!-- more -->
遇到这种问题最好的办法是查命令帮助和官方文档
```
tensorboard --helpful
可以看到命令行参数 --host的作用
--host ADDR           What host to listen to (default: localhost). To serve to the entire local network on both IPv4 and IPv6, see
                        `--bind_all`, with which this option is mutually exclusive.
```
所以
```
tensorboard --logdir logs --host 0.0.0.0
```
即可远程访问tensorboard