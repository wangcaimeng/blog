---
title: Visual Studio Code LaTex Workshop 修改配置默认使用xelatex编译
date: 2020-03-14 13:36:11
categories: 环境搭建
tags: latex, vscode
---
vscode安装Latex Workshop插件后可以，默认使用pdftex编译，导致我的某些公式不正常，但是在使用xelatex编译是正常的。而设置中的Latex Workshop没有直接可以修改选项，需要参考官方文档<https://github.com/James-Yu/LaTeX-Workshop/wiki/Compile#latex-recipe>修改settings.json

<!-- more -->

## 1添加编译方式

将以下配置添加到settings.json中

```json
"latex-workshop.latex.tools": [
        {
            "name": "xelatex",
            "command": "xelatex",
            "args": [
              "-synctex=1",
              "-interaction=nonstopmode",
              "-file-line-error",
              "%DOC%"
            ]
        },
        {
            "name": "pdflatex",
            "command": "pdflatex",
            "args": [
              "-synctex=1",
              "-interaction=nonstopmode",
              "-file-line-error",
              "%DOC%"
            ]
        },
        {
            "name": "bibtex",
            "command": "bibtex",
            "args": [
              "%DOCFILE%"
            ]
        }
    ]
```

## 2 添加编译组合

``` json
"latex-workshop.latex.recipes": [
        {
          "name": "XeLaTeX",
          "tools": [
            "xelatex"
          ]
        },
        {
          "name": "PDFLaTeX",
          "tools": [
            "pdflatex"
          ]
        }, 
        {
          "name": "latexmk",
          "tools": [
            "latexmk"
          ]
        },
        {
          "name": "BibTeX",
          "tools": [
            "bibtex"
          ]
        },
        {
          "name": "pdflatex -> bibtex -> pdflatex*2",
          "tools": [
            "pdflatex",
            "bibtex",
            "pdflatex",
            "pdflatex"
          ]
        },
        {
          "name": "xelatex -> bibtex -> xelatex*2",
          "tools": [
            "xelatex",
            "bibtex",
            "xelatex",
            "xelatex"
          ]
        }
    ]
```