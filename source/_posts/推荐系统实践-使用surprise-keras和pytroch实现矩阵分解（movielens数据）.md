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
      <th>4711</th>
      <td>28</td>
      <td>60684</td>
      <td>4.0</td>
      <td>1240817508</td>
    </tr>
    <tr>
      <th>87902</th>
      <td>567</td>
      <td>51077</td>
      <td>1.0</td>
      <td>1525289538</td>
    </tr>
    <tr>
      <th>33213</th>
      <td>226</td>
      <td>1923</td>
      <td>2.5</td>
      <td>1095662842</td>
    </tr>
    <tr>
      <th>43152</th>
      <td>288</td>
      <td>56339</td>
      <td>3.5</td>
      <td>1216220492</td>
    </tr>
    <tr>
      <th>17939</th>
      <td>112</td>
      <td>508</td>
      <td>4.5</td>
      <td>1513989990</td>
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
svd.fit(train_data)
predictions = svd.test(test_df[['userId','movieId','rating']].values)
#评估
accuracy.rmse(predictions, verbose=True)
```

    RMSE: 0.8807





    0.8806579563446105



# 3.使用keras实现矩阵分解


```python
import tensorflow as tf
from tensorflow.keras.layers import Input,Embedding,Reshape,Dot
from tensorflow.keras import Model
```


```python
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

#Dot计算rating
y = Dot(axes=1)([moive_vec,user_vec])

model = Model(inputs = [user_id_input,movie_id_input],outputs=y)
```


```python
model.summary()
```

    Model: "model"
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
    reshape (Reshape)               (None, 10)           0           movie_embedded[0][0]             
    __________________________________________________________________________________________________
    reshape_1 (Reshape)             (None, 10)           0           user_embedded[0][0]              
    __________________________________________________________________________________________________
    dot (Dot)                       (None, 1)            0           reshape[0][0]                    
                                                                     reshape_1[0][0]                  
    ==================================================================================================
    Total params: 103,340
    Trainable params: 103,340
    Non-trainable params: 0
    __________________________________________________________________________________________________



```python
#训练和评估
model.compile(loss='mse',optimizer='adam')
model.fit([train_df['userId'],train_df['movieId']],
          train_df['rating'],
          batch_size=256,
         epochs=20,
      validation_split=0.1)
predictions = model.predict([test_df['userId'],test_df['movieId']])
```

    WARNING:tensorflow:Falling back from v2 loop because of error: Failed to find data adapter that can handle input: (<class 'list'> containing values of types {"<class 'pandas.core.series.Series'>"}), <class 'NoneType'>
    Train on 72601 samples, validate on 8067 samples
    Epoch 1/20
    72601/72601 [==============================] - 2s 23us/sample - loss: 0.9517 - val_loss: 1.4929
    Epoch 2/20
    72601/72601 [==============================] - 2s 21us/sample - loss: 0.8698 - val_loss: 1.4361
    Epoch 3/20
    72601/72601 [==============================] - 1s 20us/sample - loss: 0.8053 - val_loss: 1.3950
    Epoch 4/20
    72601/72601 [==============================] - 2s 21us/sample - loss: 0.7555 - val_loss: 1.3636
    Epoch 5/20
    72601/72601 [==============================] - 2s 21us/sample - loss: 0.7162 - val_loss: 1.3399
    Epoch 6/20
    72601/72601 [==============================] - 2s 21us/sample - loss: 0.6848 - val_loss: 1.3216
    Epoch 7/20
    72601/72601 [==============================] - 2s 21us/sample - loss: 0.6588 - val_loss: 1.3082
    Epoch 8/20
    72601/72601 [==============================] - 2s 21us/sample - loss: 0.6379 - val_loss: 1.2975
    Epoch 9/20
    72601/72601 [==============================] - 1s 21us/sample - loss: 0.6198 - val_loss: 1.2884
    Epoch 10/20
    72601/72601 [==============================] - 2s 21us/sample - loss: 0.6047 - val_loss: 1.2815
    Epoch 11/20
    72601/72601 [==============================] - 2s 22us/sample - loss: 0.5917 - val_loss: 1.2772
    Epoch 12/20
    72601/72601 [==============================] - 2s 21us/sample - loss: 0.5807 - val_loss: 1.2720
    Epoch 13/20
    72601/72601 [==============================] - 2s 22us/sample - loss: 0.5709 - val_loss: 1.2697
    Epoch 14/20
    72601/72601 [==============================] - 2s 21us/sample - loss: 0.5626 - val_loss: 1.2670
    Epoch 15/20
    72601/72601 [==============================] - 2s 21us/sample - loss: 0.5551 - val_loss: 1.2648
    Epoch 16/20
    72601/72601 [==============================] - 2s 21us/sample - loss: 0.5480 - val_loss: 1.2634
    Epoch 17/20
    72601/72601 [==============================] - 1s 21us/sample - loss: 0.5417 - val_loss: 1.2633
    Epoch 18/20
    72601/72601 [==============================] - 2s 21us/sample - loss: 0.5361 - val_loss: 1.2625
    Epoch 19/20
    72601/72601 [==============================] - 2s 21us/sample - loss: 0.5308 - val_loss: 1.2627
    Epoch 20/20
    72601/72601 [==============================] - 2s 21us/sample - loss: 0.5258 - val_loss: 1.2614
    WARNING:tensorflow:Falling back from v2 loop because of error: Failed to find data adapter that can handle input: (<class 'list'> containing values of types {"<class 'pandas.core.series.Series'>"}), <class 'NoneType'>



```python
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_pred=predictions, y_true=test_df['rating']))
print('RMSE:',rmse)
```

    RMSE: 1.1353689899680335


# 3.使用pytroch实现矩阵分解


```python
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler,RandomSampler
```


```python
class MF(nn.Module):
    def __init__(self,user_size,movie_size):
        super(MF,self).__init__()
        self.user_embedding = nn.Embedding(user_size,embedding_dim=10)
        self.movie_embedding = nn.Embedding(movie_size,embedding_dim=10)
    
    def forward(self,user_id,movie_id):
        #embedding
        user_embedded = self.user_embedding(user_id)
        movie_embedded = self.user_embedding(user_id)
        
        ## 这里需要注意，考虑batch的存在
        ## 这里的操作是两个矩阵对应位置相乘， 然后根据列求和，即两个矩阵对应的行做dot
        out =torch.sum(user_embedded*movie_embedded,1)
        return out
```


```python
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
mf = MF(n_users,n_movies).to(device)
op = torch.optim.Adam(mf.parameters())
loss_func = nn.MSELoss()
```


```python
#训练
for epoch in range(20):
    for index in BatchSampler(RandomSampler(train_df),256,drop_last=False):
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

    RMSE: 0.9522425921132694


