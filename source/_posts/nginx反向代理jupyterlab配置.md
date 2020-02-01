---
title: nginx反向代理jupyterlab配置
date: 2020-01-31 21:18:55
categories: 环境搭建
---
最近赶上疫情，没法回学校，实验室的服务器没有公网ip，于是在出口服务器上申请一个端口反向代理到服务器的一个端口，再再服务器上使用nginx反向代理多个人的jupyterlab，以下配置可满足一个端口映射到多个jupyter-lab，供多人在学校内网的服务器上的自己的环境上开发。

<!-- more -->

## 1.基础配置
- 机器上装有nginx，并配置好监听某个端口，这里以3080端口为例
```
server {
        listen 3080;
}

```
- 配置好jupyter-lab，可以直接通过端口访问。

## 2.为每个人的jupyter-lab配置反响代理

### 2.1 jupyter-lab配置
``` python
c.NotebookApp.allow_origin = '*'
c.NotebookApp.base_url = '/$path'  #$path是自己想的一个名字，用来区分不同用户
```

### 2.2 nginx反向代理配置

```
server {
        listen 3080;

        ## 每个用户不同的$path 分别配置一个反响代理，除proxy_pass外其他选项也要配置，否则使用时会有问题
        location /jupyterlab_wcm {
                proxy_pass http://127.0.0.1:8891;
                proxy_set_header      Host $host;
                # websocket support
                proxy_http_version    1.1;
                proxy_set_header      Upgrade "websocket";
                proxy_set_header      Connection "Upgrade";
        }
        location /jupyterlab_hhm {
                proxy_pass http://127.0.0.1:9999;
                proxy_set_header      Host $host;
                # websocket support
                proxy_http_version    1.1;
                proxy_set_header      Upgrade "websocket";
                proxy_set_header      Connection "Upgrade";
        }

}
```