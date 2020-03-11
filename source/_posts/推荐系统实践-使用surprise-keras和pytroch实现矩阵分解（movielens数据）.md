---
title: '[推荐系统实践]使用surprise,keras和pytroch实现矩阵分解（movielens数据）'
date: 2020-03-08 15:09:59
categories:
    - 机器学习
tags: 
    - 算法实现
mathjax: true
---

分别使用surprise,keras和pytorch实现矩阵分解
<!-- more -->

# 1.数据加载及预处理


```python
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
```


```python
ratings_df = pd.read_csv('data/ml-latest-small/ratings.csv')
ratings_df.sample(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>45674</th>
      <td>302</td>
      <td>1221</td>
      <td>5.0</td>
      <td>854473396</td>
    </tr>
    <tr>
      <th>15358</th>
      <td>100</td>
      <td>1213</td>
      <td>3.5</td>
      <td>1100183731</td>
    </tr>
    <tr>
      <th>38978</th>
      <td>268</td>
      <td>1968</td>
      <td>2.0</td>
      <td>940183036</td>
    </tr>
    <tr>
      <th>85484</th>
      <td>555</td>
      <td>1405</td>
      <td>3.0</td>
      <td>978747341</td>
    </tr>
    <tr>
      <th>71398</th>
      <td>458</td>
      <td>552</td>
      <td>4.0</td>
      <td>845652992</td>
    </tr>
  </tbody>
</table>

</div>




```python
#用户和电影数量 
n_users = ratings_df.userId.unique().shape[0]
n_movies = ratings_df.movieId.unique().shape[0]

# 对userid重新编码
user_id_to_index = dict(zip(ratings_df.userId.unique(),[i for i in range(n_users)]))
movie_id_to_index = dict(zip(ratings_df.movieId.unique(),[i for i in range(n_movies)]))

ratings_df['userId'] = ratings_df['userId'].map(user_id_to_index)
ratings_df['movieId'] = ratings_df['movieId'].map(movie_id_to_index)
```


```python
#划分测试集训练集
from sklearn.model_selection import train_test_split
train_df,test_df = train_test_split(ratings_df,test_size = 0.2,random_state = 0)
```

# 2. surprise库实现svd分解


```python
from surprise import SVD
from surprise import Dataset,Reader
from surprise import accuracy
```


```python
reader = Reader(rating_scale=(1,5))
train_data = Dataset.load_from_df(train_df[['userId','movieId','rating']],reader).build_full_trainset()
```


```python
svd = SVD()
#训练和预测
svd.fit(train_data,)
predictions = svd.test(test_df[['userId','movieId','rating']].values)
#评估
accuracy.rmse(predictions, verbose=True)
```

    RMSE: 0.8807





    0.8806579563446105



# 3.使用keras实现矩阵分解

使用深度学习框架的方法，对userid和movieid进行embedding后进行Dot得到rating。还可以讲连个embedded向量进行拼接使用2个LinearLayer进行预测rating


```python
import tensorflow as tf
from tensorflow.keras.layers import Input,Embedding,Reshape,Dot,Dense,Concatenate
from tensorflow.keras import Model
import os
# 使用第2张GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
```


```python
def build_model(outlayer='dot'):
    #输入层
    user_id_input = Input(shape=[1],name='user_id')
    movie_id_input = Input(shape=[1],name='movied_id')

    #embedding 层
    embedding_dim = 10
    user_embedded = Embedding(input_dim=n_users,output_dim=embedding_dim,input_length=1,name='user_embedded')(user_id_input)
    movie_embedded = Embedding(input_dim=n_movies,output_dim=embedding_dim,input_length=1,name='movie_embedded')(movie_id_input)

    #Reshape embedding层输出是[batch,1,embedding_dim]
    moive_vec = Reshape([embedding_dim])(movie_embedded)
    user_vec = Reshape([embedding_dim])(user_embedded)

    
    if outlayer == 'dot':
        #矩阵分解 Dot计算rating
        y = Dot(axes=1)([moive_vec,user_vec])
    elif outlayer == 'fc':
        #使用全连接层预测rating 效果更好
        concat = Concatenate()([moive_vec,user_vec])
        dense = Dense(256)(concat)
        y = Dense(1)(dense)

    model = Model(inputs = [user_id_input,movie_id_input],outputs=y)
    return model
```


```python
model = build_model('fc')
model.summary()
```

```
    Model: "model_5"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    movied_id (InputLayer)          [(None, 1)]          0                                            
    __________________________________________________________________________________________________
    user_id (InputLayer)            [(None, 1)]          0                                            
    __________________________________________________________________________________________________
    movie_embedded (Embedding)      (None, 1, 10)        97240       movied_id[0][0]                  
    __________________________________________________________________________________________________
    user_embedded (Embedding)       (None, 1, 10)        6100        user_id[0][0]                    
    __________________________________________________________________________________________________
    reshape_10 (Reshape)            (None, 10)           0           movie_embedded[0][0]             
    __________________________________________________________________________________________________
    reshape_11 (Reshape)            (None, 10)           0           user_embedded[0][0]              
    __________________________________________________________________________________________________
    concatenate_3 (Concatenate)     (None, 20)           0           reshape_10[0][0]                 
                                                                     reshape_11[0][0]                 
    __________________________________________________________________________________________________
    dense_6 (Dense)                 (None, 256)          5376        concatenate_3[0][0]              
    __________________________________________________________________________________________________
    dense_7 (Dense)                 (None, 1)            257         dense_6[0][0]                    
    ==================================================================================================
    Total params: 108,973
    Trainable params: 108,973
    Non-trainable params: 0
    __________________________________________________________________________________________________

```

```python
#训练和评估
model.compile(loss='mse',optimizer='adam')
model.fit([train_df['userId'],train_df['movieId']],
          train_df['rating'],
          batch_size=256,
         epochs=20)
predictions = model.predict([test_df['userId'],test_df['movieId']])
```


```python
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_pred=predictions, y_true=test_df['rating']))
print('RMSE:',rmse)
```



# 4.使用pytroch实现矩阵分解


```python
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler,RandomSampler
```


```python
class MF(nn.Module):
    def __init__(self,user_size,movie_size,embedding_dim,outlayer='dot'):
        super(MF,self).__init__()
        self.outlayer = outlayer
        self.user_embedding = nn.Embedding(user_size,embedding_dim=embedding_dim)
        self.movie_embedding = nn.Embedding(movie_size,embedding_dim=embedding_dim)
        if outlayer == 'fc':
            self.fc1 = nn.Linear(in_features=2*embedding_dim,out_features=256)
            self.fc2 = nn.Linear(in_features=256,out_features=1)
    
    def forward(self,user_id,movie_id):
        #embedding
        user_embedded = self.user_embedding(user_id)
        movie_embedded = self.movie_embedding(user_id)
        if self.outlayer == 'dot':
            ## 这里需要注意，考虑batch的存在
            ## 这里的操作是两个矩阵对应位置相乘， 然后根据列求和，即两个矩阵对应的行做dot
            out =torch.sum(user_embedded*movie_embedded,1)
        elif self.outlayer == 'fc':
            concat = torch.cat((user_embedded,movie_embedded),1)
            x = self.fc1(concat)
            x = torch.relu(x)
            out = self.fc2(x)
        return out
```


```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mf = MF(n_users,n_movies,10,'fc').to(device)
op = torch.optim.Adam(mf.parameters())
loss_func = nn.MSELoss()
```


```python
#训练

for epoch in tqdm_notebook(range(20),desc='整体进度'):
    for index in tqdm_notebook(BatchSampler(RandomSampler(train_df),256,drop_last=True),desc='第{}个epoch0'.format(epoch),leave=False):
        batch = train_df.iloc[index]
        user_id = torch.tensor(batch['userId'].tolist()).to(device)
        movie_id = torch.tensor(batch['movieId'].tolist()).to(device)
        predict = mf(user_id,movie_id)
        loss = loss_func(predict,torch.tensor(batch['rating'].tolist()).to(device))
        op.zero_grad()
        loss.backward()
        op.step()
        
```

```python
#预测
predictions = mf(torch.as_tensor(test_df['userId'].to_numpy()).to(device),torch.as_tensor(test_df['movieId'].to_numpy()).to(device))
```


```python
rmse = np.sqrt(mean_squared_error(y_pred=predictions.detach().cpu().numpy(), y_true=test_df['rating']))
print('RMSE:',rmse)
```





