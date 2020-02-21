---
title: 使用docker部署flask+vue.js前后端分离webapp
date: 2019-12-13 14:26:01
categories: 全栈技术
tags:
---

本文主要讲解如何将前后端分离的web应用，分别构建成docker镜像并部署。

<!-- more -->

# 1. 后端

## 1.1 目录结构


```
back_end_docker
├── lrrs-back-end //我项目的名字
│   ├── app.py  //flask入口文件
│   ├── requirements.txt  //python依赖
│   └── uwsgi.ini   //uwsgi配置文件
├── Dockerfile   
└── nginx.conf  //nginx配置文件
```

## 1.2 Dockerfile

``` Dockerfile
# 配置基础镜像  这里使用ubuntu  之前使用alpine 安装numpy出现问题
FROM ubuntu:18.04

# 添加标签说明
LABEL author="wangcaimeng" email="wangcaimeng@nlsde.buaa.edu.cn"  purpose="nginx+uwsgi+python3"

# 配置apt镜像源地址  使用阿里云镜像站
RUN  sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN  apt-get clean
RUN apt-get update

# ubuntu镜像默认没有  不添加python3会报编码错误（即使python3 默认utf-8编码）
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8


# 安装软件
RUN apt-get install -y  nginx  python3-pip python3-dev uwsgi-plugin-python3  uwsgi 



# 建立软链接
RUN ln -s /usr/bin/python3 /usr/bin/python

# 安装依赖

# 创建工作路径
RUN mkdir /app

# 指定容器启动时执行的命令都在app目录下执行
WORKDIR /app

# 替换nginx的配置
COPY nginx.conf /etc/nginx/nginx.conf

# 将本地app目录下的内容拷贝到容器的app目录下       注意操作的文件不能再dockerfile的外层
COPY ./lrrs-back-end/ /app/

RUN pip3 install -r /app/requirements.txt  -i  https://pypi.tuna.tsinghua.edu.cn/simple  --no-cache-dir

# 启动nginx和uwsgi
ENTRYPOINT nginx -g "daemon on;" && uwsgi --ini /app/uwsgi.ini

```

## 1.3 nginx配置
参考https://www.cnblogs.com/beiluowuzheng/p/10220860.html

```conf
user  root;
worker_processes  1;
error_log  /var/log/nginx/error.log warn;
pid        /var/run/nginx.pid;
worker_rlimit_nofile 20480;


events {
  use epoll;
  worker_connections 20480;
  multi_accept on;
}


http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;

    log_format  main  '$remote_addr - $remote_user [$time_local] "$request" '
                      '$status $body_bytes_sent "$http_referer" '
                      '"$http_user_agent" "$http_x_forwarded_for"';
    #请求量级大建议关闭acccess_log
    #access_log  /var/log/nginx/access.log  main;

    sendfile        on;
    #tcp_nopush     on;

    keepalive_timeout  300s;
    client_header_timeout 300s;
    client_body_timeout 300s;

    gzip on;
    gzip_min_length 1k;
    gzip_buffers 4 16k;
    gzip_types text/html application/javascript application/json;

    include /etc/nginx/conf.d/*.conf;

    server {
      listen 6666;
      charset utf-8;
      client_max_body_size 75M;
      location / {
        include uwsgi_params;
        uwsgi_pass unix:///tmp/uwsgi.sock;
        uwsgi_send_timeout 300;
        uwsgi_connect_timeout 300;
        uwsgi_read_timeout 300;
      }
    }
}

```

## 1.4 uwsgi配置
[uwsgi]
uwsgi-socket    = /tmp/uwsgi.sock
chmod-socket    = 777
callable        = app
plugin          = python3   
wsgi-file       = app.py
buffer-size     = 65535
processes       = 2
threads         = 20
disable-logging = true

# 2. 前端

## 2.1目录结构

```
front_end_docker
├── lrrs-front-end //我项目的名字
│   ├── dist  //build好的项目
│   └── ...
├── Dockerfile   
└── nginx
    └── default.conf
```

nginx.conf  //nginx配置文件

## 2.1 Dockerfile
``` Dockerfile
#使用官方的nginx镜像
FROM nginx  

# 复制文件
## 这里需要注意 我这里通过子路径访问 即www.xxx.com/lrrs/访问到的是我的项目 nginx和vue中需要进行配置  vue中需要注意url的mode
COPY lrrs-front-end/dist/ /usr/share/nginx/html/lrrs/  
COPY nginx/default.conf /etc/nginx/conf.d/default.conf
```

## 2.2 Nginx配置
```conf
server {
listen       80;
server_name  localhost;

#charset koi8-r;
access_log  /var/log/nginx/host.access.log  main;
error_log  /var/log/nginx/error.log  error;

# 这里需要注意 我这里通过子路径访问 即www.xxx.com/lrrs/访问到的是我的项目 nginx和vue中需要进行配置 
location /lrrs {
    root   /usr/share/nginx/html;
    index  index.html index.htm;
}

#error_page  404              /404.html;

# redirect server error pages to the static page /50x.html
#
error_page   500 502 503 504  /50x.html;
location = /50x.html {
    root   /usr/share/nginx/html;
}
} 

```

# 3. 调试技巧

直接使用dockerfile build如果无法使用，建议先pull基础镜像 进入基础镜像手动配置并调试 调试好后再写dockerfile