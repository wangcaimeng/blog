---
title: hexo配置latex数学公式支持
date: 2019-10-25 22:31:45
categories: 环境搭建
tags:
mathjax: true
---

我使用的是最新的hexo4.0。默认无法进行letex数学公式渲染，查阅一番资料，主要有以下解决方案：

- 使用hexo-math插件
- 使用Next主题
  
<!-- more -->

# 1.使用hexo-math插件
## 1.1安装hexo-math插件
https://github.com/hexojs/hexo-math
``` shell
npm install hexo-math --save
```

## 1.2 修改配置配置文件
在站点的_config.yml中添加：
``` yaml
math:
  engine: 'mathjax' # or 'katex'
  mathjax:
    src: https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML
    config: {
      tex2jax: {
        inlineMath: [ ['$','$'], ["\\(","\\)"] ],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
        processEscapes: true
      },
      TeX: {
        equationNumbers: {
          autoNumber: "AMS"
        }
      }
    }
```
注意config=TeX-AMS-MML_HTMLorMML很关键，没有的话会渲染失败。

修改编辑node_modules\marked\lib\marked.js 
```
搜索escape: 
替换为
escape: /^\\([`*\[\]()# +\-.!_>])/,
这一步取消了对\\,\{,\}的转义(escape)
搜索em:/^\b_((?:[^_]|__)+?)_\b|^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,
替换为
em:/^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,
这一步取消了对斜体标记_的转义
```

在md文件头部添加：mathjax: true，如：
```
---
title: hexo配置latex数学公式支持
date: 2019-10-25 22:31:45
categories: 环境搭建
tags:
mathjax: true
---
```

## 1.3 测试
$$ a^{2}=b^{2}+c^{2} $$

# 2 使用next主题

<https://github.com/theme-next/hexo-theme-next>
参考<https://blog.csdn.net/qq_41518277/article/details/101766036>

- 使用mathjax
  卸载hexo-renderer-marked并更换为一下二选一
  ```
    npm un hexo-renderer-marked --save 
    npm i hexo-renderer-kramed --save ＃ or hexo-renderer-pandoc
  ```
  在主题的配置文件中修改
  ```
  mathjax：    
  enable：true
  ```
- 使用katex
  卸载hexo-renderer-marked并更换为一下二选一
  ```
  npm un hexo-renderer-marked --save 
  npm i hexo-renderer-markdown-it-plus --save ＃ or hexo-renderer-markdown-it
  ```
  在主题的配置文件中修改
  ```
  katex:
  enable：true
  ```