---
title: 记第二次面试——作业帮算法实习（已收到offer）
date: 2019-03-09 12:06:34
categories:
    - 面试
tags: 
    - 实习面试
    - 作业帮
---
## 0.前言
周一面试头条失利之后回来做了些总结。这一周继续刷题并把做过的题总结了一遍，把《统计学习方法》上经典的算法再手推了一下，还看了GDBT和XGBoost这两个比较火的算法，都手动推导了。然后投了一些公司，周4收到作业帮，百度和创新奇智
我在学校主要是做AI+教育方向的，作业帮这个岗位也是做学生能力评估、学习资源推荐，和我我在学校实验室做的非常match。

<!-- more -->

## 1.面试
### 1.1 一面

 - 首先是介绍自己的项目，我做的是基于强化学习的学习资源推荐，面试官非常感兴趣，给他详细讲了自己的做的和基于的别人的成果。
 - 接下来是第一道算法题，要手写代码，乘积小于K的连续子数组https://leetcode-cn.com/problems/subarray-product-less-than-k/
面试官说可以先写个暴力的看看你的代码再做优化。写完后直接出了第二道题
- 第二道，平面直接坐标系给一些了点对，找出其组成的最小矩形的面积，若没有矩形返回0。如果4个点4层循环遍历要O(n^ 4)，然后面试官提示我如果2组点对的中点相同且距离相等，那么他们可以构成一个长方形。然后写代码实现，两层循环遍历，用一个字典（python的hashmap）存信息，key是(mid，distance)，value存这个点对的索引(i,j)，遍历的时候如果key不存在就存进去，存在就说明这个点对和已经存在的点对构成一个矩形，更新最小面积即可。复杂度O(n^2)。类似于这题https://leetcode-cn.com/problems/minimum-area-rectangle/，但是矩形的边不一定是和x，y轴平行的。

### 1.2 二面
- 先问了一下项目中点问题，问我能想到什么解决方法，感觉主要考察用机器学习问题的思想，但是也没有问机器学习算法。
- 然后又是手写代码，第一道题我刚好搜别人的面试总结时见过， 就是这个题 https://blog.csdn.net/program_developer/article/details/79486088 我的思路是按起始节点先排序，然后合并s1，计算总长度，然后合并后的结果中加入s2的线段再排序再合并，再遍历计算长度，如果长度没变，说明覆盖了，如果变了返回False。面试官问了我复杂度，我说就是排序的复杂度，剩下的操作都是线性的复杂度。然后让我写代码，这回不让我写python了。。说你简历上写着还会java和c 你选一个，然后用java写的，很久没写java了，在纸上写还真有点儿生疏。。
- 第二道是升级版的题目，面试官说这个就说说思路，不用写代码了。平面直角坐标系上有一组长方形，边和坐标轴平行，计算总面积。从左到右扫描，需要考虑的就是什么时候算面积，回答又不少考虑不周的情况，最后面试官跟我说了他的思路，最后算法会用到第一题的东西。这道体感觉是不会影响结果的那种。接下来让我问问题，我问了他们组主要做什么，有没有上线。然后他说联系hr来聊，我就美滋滋了。

### 1.3 HR面
hr问了问学校实验室的情况，时间情况。我说能安排好时间。他说为什么研一就出来实习，我说想让自己紧张点儿，多学点儿东西，提高就业竞争力。 然后跟我说了工作时间和待遇。一周3天以上，10点上班弹性工作，管三餐，日薪也算是互联网公司中比较高的。整体条件很不错，做的方向也很相关，地点再西二旗，离知春路地铁20分钟，走路10分钟，也很近，10点上班也不用挤早高峰地铁。所以我直接答应了，晚上直接发了offer，然后我就把周一面试推掉了。


