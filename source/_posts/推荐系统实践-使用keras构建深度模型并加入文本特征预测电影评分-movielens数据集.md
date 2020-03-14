---
title: '[推荐系统实践]使用keras构建深度模型并加入文本特征预测电影评分(movielens数据集)'
date: 2020-03-14 13:21:54
categories:
    - 机器学习
tags: 
    - 算法实现
mathjax: true
---

上篇文章值仅使用用户给电影评分的数据（rating.csv），使用矩阵分解的方法对评分进行了预测。这篇文章使用深度模型，并加入了电影标题和类别特征。（模型架构仅仅是个Demo，没有进行精心设计和调参）

<!-- more -->

```python
import pandas as pd
import numpy as np
import scipy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
```

## 1 数据处理

- 读取数据


```python
TEST = True
path = 'data/ml-latest-small/' if TEST else 'data/ml-25m/'
movies_df = pd.read_csv(path+'movies.csv')
movies_df.sample(5)
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
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>34955</th>
      <td>146638</td>
      <td>The forbidden education (2012)</td>
      <td>Comedy|Drama</td>
    </tr>
    <tr>
      <th>40400</th>
      <td>159333</td>
      <td>The Liverpool Goalie (2010)</td>
      <td>Children|Comedy</td>
    </tr>
    <tr>
      <th>48637</th>
      <td>177085</td>
      <td>Jagat (2015)</td>
      <td>Crime|Drama</td>
    </tr>
    <tr>
      <th>42759</th>
      <td>164568</td>
      <td>Interrogation (2016)</td>
      <td>Action|Thriller</td>
    </tr>
    <tr>
      <th>7386</th>
      <td>7625</td>
      <td>Girl (1998)</td>
      <td>Drama</td>
    </tr>
  </tbody>
</table>
</div>




```python
## 这部分数据比较少 暂时没有使用
tags_df = pd.read_csv(path+'tags.csv')
tags_df.sample(5)
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
      <th>tag</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>335510</th>
      <td>23461</td>
      <td>190505</td>
      <td>intimate</td>
      <td>1559946325</td>
    </tr>
    <tr>
      <th>249697</th>
      <td>13517</td>
      <td>133645</td>
      <td>photography</td>
      <td>1488389255</td>
    </tr>
    <tr>
      <th>764421</th>
      <td>101979</td>
      <td>968</td>
      <td>cult film</td>
      <td>1568376838</td>
    </tr>
    <tr>
      <th>1047854</th>
      <td>155482</td>
      <td>4085</td>
      <td>on race</td>
      <td>1486161395</td>
    </tr>
    <tr>
      <th>60017</th>
      <td>6550</td>
      <td>2932</td>
      <td>class differences</td>
      <td>1528592296</td>
    </tr>
  </tbody>
</table>
</div>




```python
ratings_df = pd.read_csv(path+'ratings.csv')
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
      <th>16208353</th>
      <td>105082</td>
      <td>8012</td>
      <td>3.5</td>
      <td>1094070343</td>
    </tr>
    <tr>
      <th>24992620</th>
      <td>162516</td>
      <td>1437</td>
      <td>4.0</td>
      <td>1175648959</td>
    </tr>
    <tr>
      <th>13913658</th>
      <td>90192</td>
      <td>2018</td>
      <td>3.5</td>
      <td>1256591385</td>
    </tr>
    <tr>
      <th>21678127</th>
      <td>140907</td>
      <td>2208</td>
      <td>4.0</td>
      <td>942164194</td>
    </tr>
    <tr>
      <th>20925384</th>
      <td>136085</td>
      <td>195</td>
      <td>3.0</td>
      <td>860170604</td>
    </tr>
  </tbody>
</table>
</div>



- 处理genres列并或获取所有的genres


```python
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
```


```python
# genres转小写，分割成为数组
movies_df['genres'] = movies_df['genres'].str.lower().str.split('|')
genres = movies_df['genres'].tolist()
genres  = list({g for list_ in genres for g in list_  })
genres
```




    ['war',
     '(no genres listed)',
     'thriller',
     'action',
     'animation',
     'crime',
     'sci-fi',
     'imax',
     'romance',
     'horror',
     'adventure',
     'film-noir',
     'mystery',
     'children',
     'documentary',
     'musical',
     'fantasy',
     'western',
     'comedy',
     'drama']



- 年份数据


```python
movies_df['year'] = movies_df['title'].str.extract('.*\(([0-9]*)\).*')
movies_df['title'] = movies_df['title'].str.replace('(\([0-9]*\))','').str.lower()
movies_df.sample(5)
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
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>53412</th>
      <td>187369</td>
      <td>arrivée d'un train à perrache</td>
      <td>[documentary]</td>
      <td>1896</td>
    </tr>
    <tr>
      <th>40930</th>
      <td>160569</td>
      <td>ice age: collision course</td>
      <td>[adventure, animation, children, comedy]</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>20137</th>
      <td>104314</td>
      <td>hitler's children</td>
      <td>[documentary]</td>
      <td>2011</td>
    </tr>
    <tr>
      <th>10254</th>
      <td>37744</td>
      <td>story of vernon and irene castle, the</td>
      <td>[musical, romance, war]</td>
      <td>1939</td>
    </tr>
    <tr>
      <th>59822</th>
      <td>201715</td>
      <td>path of blood</td>
      <td>[documentary]</td>
      <td>2018</td>
    </tr>
  </tbody>
</table>
</div>



- 处理tag列


```python
tags_df['tag'] = tags_df['tag'].str.lower()
tags = tags_df['tag'].unique()
tags
```




    array(['classic', 'sci-fi', 'dark comedy', ..., 'genre busting',
           'the wife did it', 'cornetto triolgy'], dtype=object)



- 拼接成为一个宽表


```python
data_df = ratings_df.set_index(['movieId','userId']).join(movies_df.set_index('movieId')).join(tags_df.set_index(['movieId','userId']),lsuffix='_rating',rsuffix='_taging')
```


```python
data_df = data_df.reset_index()
data_df
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
      <th>movieId</th>
      <th>userId</th>
      <th>rating</th>
      <th>timestamp_rating</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
      <th>tag</th>
      <th>timestamp_taging</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3.5</td>
      <td>1141415820</td>
      <td>toy story</td>
      <td>[adventure, animation, children, comedy, fantasy]</td>
      <td>1995</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3</td>
      <td>4.0</td>
      <td>1439472215</td>
      <td>toy story</td>
      <td>[adventure, animation, children, comedy, fantasy]</td>
      <td>1995</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>4</td>
      <td>3.0</td>
      <td>1573944252</td>
      <td>toy story</td>
      <td>[adventure, animation, children, comedy, fantasy]</td>
      <td>1995</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>5</td>
      <td>4.0</td>
      <td>858625949</td>
      <td>toy story</td>
      <td>[adventure, animation, children, comedy, fantasy]</td>
      <td>1995</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>8</td>
      <td>4.0</td>
      <td>890492517</td>
      <td>toy story</td>
      <td>[adventure, animation, children, comedy, fantasy]</td>
      <td>1995</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>25624096</th>
      <td>209157</td>
      <td>119571</td>
      <td>1.5</td>
      <td>1574280748</td>
      <td>we</td>
      <td>[drama]</td>
      <td>2018</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25624097</th>
      <td>209159</td>
      <td>115835</td>
      <td>3.0</td>
      <td>1574280985</td>
      <td>window of the soul</td>
      <td>[documentary]</td>
      <td>2001</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25624098</th>
      <td>209163</td>
      <td>6964</td>
      <td>4.5</td>
      <td>1574284913</td>
      <td>bad poems</td>
      <td>[comedy, drama]</td>
      <td>2018</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25624099</th>
      <td>209169</td>
      <td>119571</td>
      <td>3.0</td>
      <td>1574291826</td>
      <td>a girl thing</td>
      <td>[(no genres listed)]</td>
      <td>2001</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25624100</th>
      <td>209171</td>
      <td>119571</td>
      <td>3.0</td>
      <td>1574291937</td>
      <td>women of devil's island</td>
      <td>[action, adventure, drama]</td>
      <td>1962</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>25624101 rows × 9 columns</p>
</div>




```python
users = data_df['userId'].unique()
movies = data_df['movieId'].unique()
users_map = {v:i for i,v in enumerate(users)}
movies_map = {v:i for i,v in enumerate(movies)}
```


```python
data_df['userId'] = data_df['userId'].map(users_map)
data_df['movieId'] = data_df['movieId'].map(movies_map)
```

- 处理文本信息，padding成为一样长度


```python
genresTokenizer = Tokenizer()
genresTokenizer.fit_on_texts(data_df['genres'])
genres_seqs = genresTokenizer.texts_to_sequences(data_df['genres'])
genres_seqs = sequence.pad_sequences(genres_seqs)
```


```python
titleTokenizer = Tokenizer()
titleTokenizer.fit_on_texts(data_df['title'])
title_seqs = titleTokenizer.texts_to_sequences(data_df['title'])
title_seqs = sequence.pad_sequences(title_seqs)
```


```python
titles = titleTokenizer.word_index.keys()
```

- 划分训练集测试集


```python
from sklearn.model_selection import train_test_split
train_index,test_index = train_test_split(range(len(data_df)),test_size=0.2,random_state=0)
```

## 2 使用Keras实现深度学习模型预测rating


```python
import tensorflow as tf
from tensorflow.keras.layers import Input,Embedding,Dense,Reshape,Flatten,Concatenate,Conv1D,MaxPool1D
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
```


```python
def build_model():
    # Input Layer
    user_id_input = Input(shape=[1],name='user_id_input')
    movied_id_input = Input(shape=[1],name='movie_id_input')
    genres_input = Input(shape=[len(genres_seqs[0])],name='genres_input')
    title_input = Input(shape=[len(title_seqs[0])],name='title_len')
    
    # Embedding Layer
    user_embedded = Embedding(output_dim=10,input_dim=len(users),input_length=1)(user_id_input)
    movie_embedded = Embedding(output_dim=10,input_dim=len(movies),input_length=1)(movied_id_input)
    
    genres_embedded = Embedding(output_dim=50,input_dim=len(genres)+1,input_length=len(genres_seqs[0]))(genres_input)
    title_embedded = Embedding(output_dim=200,input_dim=len(titles)+1,input_length=len(title_seqs[0]))(title_input)
    
    #Reshape [batch_size,1,embedding_size] to [batch_size,embedding_size] 
    user_embedded = Reshape([10])(user_embedded)
    movie_embedded = Reshape([10])(movie_embedded)
    

    ## title 和genres embedding后是(batch*length*out*dim), 可以直接展开 或者用1D CNN
    #Flatten text embedded
#     genres_embedded = Flatten()(genres_embedded)
#     title_embedded = Flatten()(title_embedded)
#     text_vec = Concatenate()([genres_embedded,title_embedded])
    
    #CNN    
    genres_conv = Conv1D(32,5,strides=1)(genres_embedded)
    genres_conv = MaxPool1D(2)(genres_conv)
    genres_conv = Conv1D(64,3,strides=1)(genres_conv)
    genres_conv = Flatten()(genres_conv)
    
    title_conv = Conv1D(32,5,strides=1)(title_embedded)
    title_conv = MaxPool1D(2)(title_conv)
    title_conv = Conv1D(64,3,strides=1)(title_conv)
    title_conv = Flatten()(title_conv)
    
    text_vec = Concatenate()([genres_conv,title_conv])
    
    
    text_vec = Dense(1024,activation='relu')(text_vec)
    text_vec = Dense(512,activation='relu')(text_vec)
    text_vec = Dense(128)(text_vec)
    
    all_vec = Concatenate()([text_vec,user_embedded,movie_embedded])
    all_vec = Dense(256,activation='relu')(all_vec)
    rating = Dense(1)(all_vec)
    model = Model(inputs=[user_id_input,movied_id_input,genres_input,title_input],outputs=rating)
    return model
```


```python
model = build_model()
plot_model(model,show_shapes=True,show_layer_names=False)
```

![png](推荐系统实践-使用keras构建深度模型并加入文本特征预测电影评分-moviedlens数据集/output_27_0.png)

```python
checkpoint = ModelCheckpoint('deep_with_moive_content.bin',
                             monitor='val_loss', 
                             verbose=1, save_best_only=True,
                            mode='max')
model.compile(loss='mse',optimizer='adam')
model.fit([data_df.iloc[train_index]['userId'],data_df.iloc[train_index]['movieId'],
           genres_seqs[train_index],title_seqs[train_index]],
          data_df.iloc[train_index]['rating'],
         epochs=10,batch_size=512,
         callbacks=[checkpoint],validation_split=0.1)
```




```python
predictions = model.predict([data_df.iloc[test_index]['userId'],
                             data_df.iloc[test_index]['movieId'],
                             genres_seqs[test_index],
                             title_seqs[test_index]])
```

```python
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_pred=predictions, y_true=data_df.iloc[test_index]['rating']))
print('RMSE:',rmse)
```

    RMSE: 0.7928375214166755


