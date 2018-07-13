
# coding: utf-8

# In[1]:

import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split



# In[2]:

print('Loading data...')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
songs = pd.read_csv('songs.csv')
members = pd.read_csv('members.csv',parse_dates=['registration_init_time','expiration_date'])
songs_extra = pd.read_csv('song_extra_info.csv')
print('Done loading...')
print("zhuan hua lei xing")
train[['msno', 'source_system_tab','source_screen_name','source_type','song_id']]=train[['msno', 'source_system_tab','source_screen_name','source_type','song_id']].apply(lambda x: x.astype('category'))
train[['target']]=train[['target']].apply(lambda x: x.astype(np.uint8))
test[['msno', 'source_system_tab','source_screen_name','source_type','song_id']]=test[['msno', 'source_system_tab','source_screen_name','source_type','song_id']].apply(lambda x: x.astype('category'))
songs[['genre_ids', 'language','artist_name' ,'composer','lyricist','song_id' ]]=songs[['genre_ids', 'language','artist_name' ,'composer','lyricist','song_id' ]].apply(lambda x: x.astype('category'))
members [['city', 'gender','registered_via' ]]=members [['city', 'gender','registered_via' ]].apply(lambda x: x.astype('category'))
members[['bd']]=members[['bd']].apply(lambda x: x.astype(np.uint8))


# In[3]:

print('Data merging...')


train = train.merge(songs, on='song_id', how='left')
test = test.merge(songs, on='song_id', how='left')

members['membership_days'] = members['expiration_date'].subtract(members['registration_init_time']).dt.days.astype(int)

members['registration_year'] = members['registration_init_time'].dt.year
members['registration_month'] = members['registration_init_time'].dt.month
members['registration_date'] = members['registration_init_time'].dt.day

members['expiration_year'] = members['expiration_date'].dt.year
members['expiration_month'] = members['expiration_date'].dt.month
members['expiration_date'] = members['expiration_date'].dt.day
members = members.drop(['registration_init_time'], axis=1)

def isrc_to_year(isrc):
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return 1900 + int(isrc[5:7])
        else:
            return 2000 + int(isrc[5:7])
    else:
        return np.nan
        
songs_extra['song_year'] = songs_extra['isrc'].apply(isrc_to_year)
songs_extra.drop(['isrc', 'name'], axis = 1, inplace = True)

train = train.merge(members, on='msno', how='left')
test = test.merge(members, on='msno', how='left')

train = train.merge(songs_extra, on = 'song_id', how = 'left')
train.song_length.fillna(200000,inplace=True)
train.song_length = train.song_length.astype(np.uint32)
train.song_id = train.song_id.astype('category')


test = test.merge(songs_extra, on = 'song_id', how = 'left')
test.song_length.fillna(200000,inplace=True)
test.song_length = test.song_length.astype(np.uint32)
test.song_id = test.song_id.astype('category')

import gc
del members, songs; gc.collect();

print('Done merging...')


# In[4]:

## Converting object types to categorical
train = pd.concat([
        train.select_dtypes([], ['object']),
        train.select_dtypes(['object']).apply(pd.Series.astype, dtype='category')
        ], axis=1).reindex_axis(train.columns, axis=1)

test = pd.concat([
        test.select_dtypes([], ['object']),
        test.select_dtypes(['object']).apply(pd.Series.astype, dtype='category')
        ], axis=1).reindex_axis(test.columns, axis=1)


# In[5]:

# In[6]:
print("nian liang")
def age(x):
    if x==0:
        return 29
    else:
        return x
train['bd'] = train['bd'].apply(age).astype(np.int16)
test['bd'] = test['bd'].apply(age).astype(np.int16)

print("count features")
def lyricist_count(x):
    if x == 'no_lyricist':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';'])) + 1

train['lyricist'] = train['lyricist'].cat.add_categories(['no_lyricist'])
train['lyricist'].fillna('no_lyricist',inplace=True)
train['lyricists_count'] = train['lyricist'].apply(lyricist_count).astype(np.int8)
test['lyricist'] = test['lyricist'].cat.add_categories(['no_lyricist'])
test['lyricist'].fillna('no_lyricist',inplace=True)
test['lyricists_count'] = test['lyricist'].apply(lyricist_count).astype(np.int8)

def composer_count(x):
    if x == 'no_composer':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';'])) + 1

train['composer'] = train['composer'].cat.add_categories(['no_composer'])
train['composer'].fillna('no_composer',inplace=True)
train['composer_count'] = train['composer'].apply(composer_count).astype(np.int8)
test['composer'] = test['composer'].cat.add_categories(['no_composer'])
test['composer'].fillna('no_composer',inplace=True)
test['composer_count'] = test['composer'].apply(composer_count).astype(np.int8)

def artist_count(x):
    if x == 'no_artist':
        return 0
    else:
        return x.count('and') + x.count(',') + x.count('feat') + x.count('&')

train['artist_count'] = train['artist_name'].apply(artist_count).astype(np.int8)
test['artist_count'] = test['artist_name'].apply(artist_count).astype(np.int8)

# number of times a song has been played before
_dict_count_song_played_train = {k: v for k, v in train['song_id'].value_counts().iteritems()}
_dict_count_song_played_test = {k: v for k, v in test['song_id'].value_counts().iteritems()}
def count_song_played(x):
    try:
        return _dict_count_song_played_train[x]
    except KeyError:
        try:
            return _dict_count_song_played_test[x]
        except KeyError:
            return 0
    

train['count_song_played'] = train['song_id'].apply(count_song_played).astype(np.int64)
test['count_song_played'] = test['song_id'].apply(count_song_played).astype(np.int64)

# number of times the artist has been played
_dict_count_artist_played_train = {k: v for k, v in train['artist_name'].value_counts().iteritems()}
_dict_count_artist_played_test = {k: v for k, v in test['artist_name'].value_counts().iteritems()}
def count_artist_played(x):
    try:
        return _dict_count_artist_played_train[x]
    except KeyError:
        try:
            return _dict_count_artist_played_test[x]
        except KeyError:
            return 0

train['count_artist_played'] = train['artist_name'].apply(count_artist_played).astype(np.int64)
test['count_artist_played'] = test['artist_name'].apply(count_artist_played).astype(np.int64)


# In[7]:

print("0,1 features")
def is_featured(x):
    if 'feat' in str(x) :
        return 1
    return 0
train['artist_name'] = train['artist_name'].cat.add_categories(['no_artist'])
train['artist_name'].fillna('no_artist',inplace=True)
train['is_featured'] = train['artist_name'].apply(is_featured).astype(np.int8)
test['artist_name'] = test['artist_name'].cat.add_categories(['no_artist'])
test['artist_name'].fillna('no_artist',inplace=True)
test['is_featured'] = test['artist_name'].apply(is_featured).astype(np.int8)

# if artist is same as composer
train['artist_composer'] = (np.asarray(train['artist_name']) == np.asarray(train['composer'])).astype(np.int8)
test['artist_composer'] = (np.asarray(test['artist_name']) == np.asarray(test['composer'])).astype(np.int8)


# if artist, lyricist and composer are all three same
train['artist_composer_lyricist'] = ((np.asarray(train['artist_name']) == np.asarray(train['composer'])) & 
                                     np.asarray((train['artist_name']) == np.asarray(train['lyricist'])) & 
                                     np.asarray((train['composer']) == np.asarray(train['lyricist']))).astype(np.int8)
test['artist_composer_lyricist'] = ((np.asarray(test['artist_name']) == np.asarray(test['composer'])) & 
                                    (np.asarray(test['artist_name']) == np.asarray(test['lyricist'])) &
                                    np.asarray((test['composer']) == np.asarray(test['lyricist']))).astype(np.int8)

# is song language 17 or 45. 
def song_lang_boolean(x):
    if '17.0' in str(x) or '45.0' in str(x):
        return 1
    return 0

train['song_lang_boolean'] = train['language'].apply(song_lang_boolean).astype(np.int8)
test['song_lang_boolean'] = test['language'].apply(song_lang_boolean).astype(np.int8)


_mean_song_length = np.mean(train['song_length'])
def smaller_song(x):
    if x < _mean_song_length:
        return 1
    return 0

train['smaller_song'] = train['song_length'].apply(smaller_song).astype(np.int8)
test['smaller_song'] = test['song_length'].apply(smaller_song).astype(np.int8)


# In[11]:


# In[8]:

print ("Train test and validation sets")
for col in train.columns:
    if train[col].dtype == object:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')
X_train = train.drop(['target'], axis=1)
y_train = train['target'].values


X_test = test.drop(['id'], axis=1)
ids = test['id'].values

train_x=X_train[:int(len(X_train)*0.8)]
val_x=X_train[int(len(X_train)*0.8):]
train_y=y_train[:int(len(y_train)*0.8)]
val_y=y_train[int(len(y_train)*0.8):]
#train_x, val_x,train_y, val_y = train_test_split(X_train, y_train,random_state=0)

del train, test; gc.collect();

lgb_train = lgb.Dataset(train_x, train_y)
lgb_val = lgb.Dataset(val_x, val_y)
print('Processed data...')


# In[12]:

params = {
        'objective': 'binary',
        'boosting': 'gbdt',
        'learning_rate': 0.3 ,
        'verbose': 0,
        'num_leaves': 200,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 256,
        'max_depth': 10,
        'metric' : 'auc'
    }

lgbm_model = lgb.train(params, train_set = lgb_train, valid_sets =lgb_val,early_stopping_rounds=100,num_boost_round=1000,verbose_eval=5)
# Verbose_eval prints output after every 5 iterations


# In[ ]:

import matplotlib.pyplot as plt
predictors  = [i for i in X_train .columns]
feat_imp = pd.Series(lgbm_model.feature_importance(importance_type='split',iteration=-1), predictors).sort_values(ascending=False)
feat_imp.to_csv('feature importances_1204_xiawu.csv')
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
plt.show()


# In[ ]:

predictions = lgbm_model.predict(X_test)
# Writing output to file
subm = pd.DataFrame()
subm['id'] = ids
subm['target'] = predictions
subm.to_csv('lgbm_submission1206_1.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')

print('Done!')

