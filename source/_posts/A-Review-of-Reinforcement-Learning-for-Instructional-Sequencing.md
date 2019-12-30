---
title: A Review of Reinforcement Learning for Instructional Sequencing
date: 2019-12-28 12:32:33
tags: 学习笔记
categories: 机器学习
mathjax: true
---

# A Review of Reinforcement Learning for Instructional Sequencing 总结

## Abstract

强化学习为优化给定学生模型的Instructional Sequencing提供了一个天然的框架，已经活跃了大约15年。但是强化学习是否真的能帮助学生学习呢？文章为了探究这个问题，对过去使用强化学习进行教学指导序列的各种尝试进行了回顾。


<!--more-->

## Introduction

简单介绍了研究的起源和历史，以及文章讨论的问题

- 1960 Ronald Howard(“father of decision analysis.”)：

  - Dynamic Programming and Markov Decision Processes : RL的基石
  - Machine-Aided Learning. ： 一篇鲜为人知的文章，指出可以使用计算机辅助制定个性化的教学顺序

- 1962 Richard Smallwood(Ronald Howard的博士学生)
  - A Decision Structure for Teaching Machines ：how to use decision processes to adapt instruction in a computerized teaching machine. This is perhaps the first example of using reinforcement learning (broadly conceived) for the purposes of instructional sequencing.

```mermaid
graph TB
       A(theory-driven+data-driven) --keep blance--- B(black-box data-driven algorithms)
```

- We find that reinforcement learning has been most successful in cases where it has been constrained with ideas and theories from cognitive psychology and the learning sciences.
- However,given that our theories and models are limited, we also find that it has been useful to complement this approach with **running more robust offline analyses  that do not rely heavily on the assumptions of one particular model**.

## Reinforcement Learning: Towards a “Theory of Instruction”

- MDP
  - $S$ is a set of states
  - $A$ is a set of actions
  - $T$ is a transition function where $T(s^\prime|s, a)$ denotes the probability of transitioning from state s to state $s^\prime$ after taking action $a$
  - $R$ is a reward function where $R(s, a)$ specifies the reward (or the probability distribution over rewards) when action a is taken in state $s$
  - $H$ is the horizon, or the number of time steps where actions are taken.
  
  In the context of instruction:
  - states are the states that hte student can be in (such as cognitive states)
  - The set of acitions are instructional activities that can change the student's cognitive state.(such as **problems, problem steps, flashcards, videos, worked examples, game levels in the context of an educational game**)

**we focus on studies where the actions are instructional activities
taken by an RL agent to optimize a student’s learning over the course of many activities.**
文中涉及的actions是由RL agent做出的，用来帮助学生学习的（如agent为学生推荐题目），而不是学生做出的（另一类问题，文中不关注）。

- OPMDP & BKT

- task-loop 
- step-loop

## A Historical Perspective

