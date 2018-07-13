import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime
import math
import gc
print('Loading data...')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print("填充train和test")
train.fillna("missing", inplace=True)
test.fillna("missing", inplace=True)

songs = pd.read_csv('songs.csv')
members = pd.read_csv('members.csv',parse_dates=['registration_init_time','expiration_date'])
songs_extra = pd.read_csv('song_extra_info.csv')
print('Done loading...')
print("加三个source特征")
train['source']=train['source_system_tab']+train['source_screen_name']+train['source_type']
test['source']=test['source_system_tab']+test['source_screen_name']+test['source_type']
# del train['source_system_tab'],train['source_screen_name'],train['source_type']
# del test['source_system_tab'],test['source_screen_name'],test['source_type']

print("转化类型")
train[['msno', 'source_system_tab','source_screen_name','source_type','song_id']]=train[['msno', 'source_system_tab','source_screen_name','source_type','song_id']].apply(lambda x: x.astype('category'))
train[['target']]=train[['target']].apply(lambda x: x.astype(np.uint8))
test[['msno', 'source_system_tab','source_screen_name','source_type','song_id']]=test[['msno', 'source_system_tab','source_screen_name','source_type','song_id']].apply(lambda x: x.astype('category'))
songs[['genre_ids', 'language','artist_name' ,'composer','lyricist','song_id' ]]=songs[['genre_ids', 'language','artist_name' ,'composer','lyricist','song_id' ]].apply(lambda x: x.astype('category'))
members [['city', 'gender','registered_via' ]]=members [['city', 'gender','registered_via' ]].apply(lambda x: x.astype('category'))
members[['bd']]=members[['bd']].apply(lambda x: x.astype(np.uint8))


print("artist总共的lang个数")
artist_lang=songs[['artist_name',  'language']].groupby(['artist_name']).agg({"language": pd.Series.nunique})
artist_lang.reset_index(inplace=True)
artist_lang.columns = list(map(''.join, artist_lang.columns.values))
artist_lang.columns = ['artist_name', 'language_count']
songs =songs.merge(artist_lang, on='artist_name')

print("artist最多的lang")
artist_lang_count=songs[['artist_name',  'language']].groupby(['artist_name',"language"],as_index=False)['language'].agg({"artist_language_count": "count"})
artist_lang_count.sort_values(['artist_name','artist_language_count'],ascending=False,inplace=True)
artist_lang_count = artist_lang_count.groupby('artist_name').head(1)
artist_lang_count=artist_lang_count[['artist_name',"language"]]
artist_lang_count.columns = ['artist_name', 'artist_main_language']
artist_lang_count=artist_lang_count.set_index('artist_name')
artist_lang_dict=artist_lang_count.to_dict()['artist_main_language']

print("artist总共的genre_ids的个数，没有什么用，待定特征")
artist_genre=songs[['artist_name',  'genre_ids']].groupby(['artist_name']).agg({"genre_ids": pd.Series.nunique})
artist_genre.reset_index(inplace=True)
artist_genre.columns = list(map(''.join, artist_genre.columns.values))
artist_genre.columns = ['artist_name', 'artist_genre_count']
songs =songs.merge(artist_genre, on='artist_name')

print("artist最多的genre,没有什么用，待定特征")
artist_genre_count=songs[['artist_name',  'genre_ids']].groupby(['artist_name',"genre_ids"],as_index=False)['genre_ids'].agg({"artist_genre_ids_count": "count"})
artist_genre_count.sort_values(['artist_name','artist_genre_ids_count'],ascending=False,inplace=True)
artist_genre_count = artist_genre_count.groupby('artist_name').head(1)
artist_genre_count=artist_genre_count[['artist_name',"genre_ids"]]
artist_genre_count.columns = ['artist_name', 'artist_main_genre_ids']
artist_genre_count=artist_genre_count.set_index('artist_name')
artist_genre_dict=artist_genre_count.to_dict()['artist_main_genre_ids']

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
songs_extra.drop(['isrc', 'name'], axis=1, inplace=True)
print("填充song year")
songs_extra.song_year.fillna(200000, inplace=True)

train = train.merge(members, on='msno', how='left')
test = test.merge(members, on='msno', how='left')

train = train.merge(songs_extra, on='song_id', how='left')
train.song_length.fillna(200000, inplace=True)
train.song_length = train.song_length.astype(np.uint32)
train.song_id = train.song_id.astype('category')

test = test.merge(songs_extra, on='song_id', how='left')
test.song_length.fillna(200000, inplace=True)
test.song_length = test.song_length.astype(np.uint32)
test.song_id = test.song_id.astype('category')

import gc
del members, songs; gc.collect();

print('Done merging...')

print("用户特征")
print("用户最喜欢的歌手 user_artist_dict")
train_user = train[['msno','artist_name']]
test_user = test[['msno','artist_name']]
df = pd.concat([train_user, test_user ], 0)
user_df=df.groupby(['msno',"artist_name"],as_index=False)['artist_name'].agg({"user_artist_count": "count"})
user_df.sort_values(['msno','user_artist_count'],ascending=False,inplace=True)
user_df = user_df.groupby('msno').head(1)
user_df=user_df[['msno',"artist_name"]]
user_df.columns = ['msno', 'msno_main_artist']
user_df=user_df.set_index('msno')
user_artist_dict=user_df.to_dict()['msno_main_artist']
print("用户最喜欢的语言 user_lang_dict")
train_user = train[['msno','language']]
test_user = test[['msno','language']]
df = pd.concat([train_user, test_user ], 0)
user_df=df.groupby(['msno',"language"],as_index=False)['language'].agg({"user_language_count": "count"})
user_df.sort_values(['msno','language'],ascending=False,inplace=True)
user_df = user_df.groupby('msno').head(1)
user_df=user_df[['msno',"language"]]
user_df.columns = ['msno', 'msno_main_language']
user_df=user_df.set_index('msno')
user_language_dict=user_df.to_dict()['msno_main_language']
print("用户最喜欢的类型 user_genre_dict")
train_user = train[['msno','genre_ids']]
test_user = test[['msno','genre_ids']]
df = pd.concat([train_user, test_user ], 0)
user_df=df.groupby(['msno',"genre_ids"],as_index=False)['genre_ids'].agg({"user_genre_ids_count": "count"})
user_df.sort_values(['msno','genre_ids'],ascending=False,inplace=True)
user_df = user_df.groupby('msno').head(1)
user_df=user_df[['msno',"genre_ids"]]
user_df.columns = ['msno', 'msno_main_genre_ids']
user_df=user_df.set_index('msno')
user_genre_ids_dict=user_df.to_dict()['msno_main_genre_ids']

del train_user,test_user,df,user_df;


print("Adding new features")


def genre_id_count(x):
    if x == 'no_genre_id':
        return 0
    else:
        return x.count('|') + 1


train['genre_ids'].fillna('no_genre_id', inplace=True)
test['genre_ids'].fillna('no_genre_id', inplace=True)
train['genre_ids_count'] = train['genre_ids'].apply(genre_id_count).astype(np.int8)
test['genre_ids_count'] = test['genre_ids'].apply(genre_id_count).astype(np.int8)


def lyricist_count(x):
    if x == 'no_lyricist':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';'])) + 1

train['lyricist'].fillna('no_lyricist', inplace=True)
test['lyricist'].fillna('no_lyricist', inplace=True)
train['lyricists_count'] = train['lyricist'].apply(lyricist_count).astype(np.int8)
test['lyricists_count'] = test['lyricist'].apply(lyricist_count).astype(np.int8)


def composer_count(x):
    if x == 'no_composer':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';'])) + 1


train['composer'].fillna('no_composer', inplace=True)
test['composer'].fillna('no_composer', inplace=True)
train['composer_count'] = train['composer'].apply(composer_count).astype(np.int8)
test['composer_count'] = test['composer'].apply(composer_count).astype(np.int8)


def is_featured(x):
    if 'feat' in str(x):
        return 1
    return 0


train['artist_name'].fillna('no_artist', inplace=True)
test['artist_name'].fillna('no_artist', inplace=True)
train['is_featured'] = train['artist_name'].apply(is_featured).astype(np.int8)
test['is_featured'] = test['artist_name'].apply(is_featured).astype(np.int8)


def artist_count(x):
    if x == 'no_artist':
        return 0
    else:
        return x.count('and') + x.count(',') + x.count('feat') + x.count('&')


train['artist_count'] = train['artist_name'].apply(artist_count).astype(np.int8)
test['artist_count'] = test['artist_name'].apply(artist_count).astype(np.int8)

# if artist is same as composer
train['artist_composer'] = (train['artist_name'] == train['composer']).astype(np.int8)
test['artist_composer'] = (test['artist_name'] == test['composer']).astype(np.int8)

print("songs with 15+ composers")
def many_composer(x):
    if x>=15:
        return 1
    else:
        return 0
train['if_many_composer'] = train['composer_count'].apply(many_composer).astype(np.int16)
test['if_many_composer'] = test['composer_count'].apply(many_composer).astype(np.int16)

print("songs with 20+ lyricist")
def many_lyricist(x):
    if x>20:
        return 1
    else:
        return 0
train['if_many_lyricist'] = train['lyricists_count'].apply(many_lyricist).astype(np.int16)
test['if_many_lyricist'] = test['lyricists_count'].apply(many_lyricist).astype(np.int16)

print('歌手最hot的语言类型')
def artist_main_lang(x):
    try:
        return artist_lang_dict[x]
    except KeyError:
        return 0
train['artist_main_lang'] = train['artist_name'].apply(artist_main_lang)
test['artist_main_lang'] = test['artist_name'].apply(artist_main_lang)
# if lang is artist main lang
train['if_artist_main_lang'] = (np.asarray(train['artist_main_lang']) == np.asarray(train['language'])).astype(np.int8)
test['if_artist_main_lang'] = (np.asarray(test['artist_main_lang']) == np.asarray(test['language'])).astype(np.int8)

print('歌手最hot的genre类型')
def artist_main_genre(x):
    try:
        return artist_genre_dict[x]
    except KeyError:
        return 0
train['artist_main_genre'] = train['artist_name'].apply(artist_main_genre)
test['artist_main_genre'] = test['artist_name'].apply(artist_main_genre)
# if lang is artist main lang
train['if_artist_main_genre'] = (np.asarray(train['artist_main_genre']) == np.asarray(train['genre_ids'])).astype(np.int8)
test['if_artist_main_genre'] = (np.asarray(test['artist_main_genre']) == np.asarray(test['genre_ids'])).astype(np.int8)

print('用户最喜欢的歌手')
def user_main_artist(x):
    try:
        return user_artist_dict[x]
    except KeyError:
        return 0
train['user_main_artist'] = train['msno'].apply(user_main_artist)
test['user_main_artist'] = test['msno'].apply(user_main_artist)
# if lang is artist main lang
train['if_user_main_artist'] = (np.asarray(train['user_main_artist']) == np.asarray(train['artist_name'])).astype(np.int8)
test['if_user_main_artist'] = (np.asarray(test['user_main_artist']) == np.asarray(test['artist_name'])).astype(np.int8)

print("用户最喜欢的语言")
def user_main_lang(x):
    try:
        return user_language_dict[x]
    except KeyError:
        return 0
train['user_main_lang'] = train['msno'].apply(user_main_lang)
test['user_main_lang'] = test['msno'].apply(user_main_lang)
# if lang is artist main lang
train['if_user_main_lang'] = (np.asarray(train['user_main_lang']) == np.asarray(train['language'])).astype(np.int8)
test['if_user_main_lang'] = (np.asarray(test['user_main_lang']) == np.asarray(test['language'])).astype(np.int8)

print("用户最喜欢的类型")
def user_main_genre(x):
    try:
        return user_genre_ids_dict[x]
    except KeyError:
        return 0
train['user_main_genre'] = train['msno'].apply(user_main_genre)
test['user_main_genre'] = test['msno'].apply(user_main_genre)
# if lang is artist main lang
train['if_user_main_genre'] = (np.asarray(train['user_main_genre']) == np.asarray(train['genre_ids'])).astype(np.int8)
test['if_user_main_genre'] = (np.asarray(test['user_main_genre']) == np.asarray(test['genre_ids'])).astype(np.int8)


# if artist, lyricist and composer are all three same
train['artist_composer_lyricist'] = (
(train['artist_name'] == train['composer']) & (train['artist_name'] == train['lyricist']) & (
train['composer'] == train['lyricist'])).astype(np.int8)
test['artist_composer_lyricist'] = (
(test['artist_name'] == test['composer']) & (test['artist_name'] == test['lyricist']) & (
test['composer'] == test['lyricist'])).astype(np.int8)


# is song language 17 or 45.
def song_lang_boolean(x):
    if '17.0' in str(x) or '45.0' in str(x) or '-1.0' in str(x):
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

print("nian liang")
def age(x):
    if x==0 or x>=70:
        return 29
    else:
        return x
train['bd'] = train['bd'].apply(age).astype(np.int16)
test['bd'] = test['bd'].apply(age).astype(np.int16)

print("Done adding features")

print ("Train test and validation sets")
for col in train.columns:
    if train[col].dtype == object:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')

#
X_train = train.drop(['target'], axis=1)
y_train = train['target'].values


X_test = test.drop(['id'], axis=1)
ids = test['id'].values

dataset_blend_train = np.zeros((X_train.shape[0], 2))
dataset_blend_test = np.zeros((X_test.shape[0], 2))
dataset_blend_test_j = np.zeros((X_test.shape[0],5))

print('K折取平均值')
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)
i=-1
for train_indices, val_indices in kf.split(train):
    i=i+1
    train_data = lgb.Dataset(train.drop(['target'], axis=1).loc[train_indices, :],
                             label=train.loc[train_indices, 'target'])
    val_data = lgb.Dataset(train.drop(['target'], axis=1).loc[val_indices, :], label=train.loc[val_indices, 'target'])

    params = {
            'objective': 'binary',
            'boosting': 'gbdt',
            'learning_rate': 0.3 ,
            'verbose': 0,
            'num_leaves': 108,
            'bagging_fraction': 0.95,
            'bagging_freq': 1,
            'bagging_seed': 1,
            'feature_fraction': 0.9,
            'feature_fraction_seed': 1,
            'max_bin': 256,
            #'max_depth': 10,
            'max_depth': 12,
            'num_rounds': 200,
            'metric' : 'auc'
        }

    bst = lgb.train(params, train_data, 200, valid_sets=[val_data],verbose_eval=5)
    y_submission=bst.predict(train.drop(['target'], axis=1).loc[val_indices, :])  #每折的预测
    dataset_blend_train[val_indices, 0] = y_submission
    dataset_blend_test_j[:, i] = bst.predict(test.drop(['id'], axis=1))
    del bst
dataset_blend_test[:,0] = dataset_blend_test_j.mean(1)
###################################################################################

dataset_blend_test_j = np.zeros((X_test.shape[0],5))

print('K折取平均值')
kf = KFold(n_splits=5)
i=-1
for train_indices, val_indices in kf.split(train):
    i=i+1
    train_data = lgb.Dataset(train.drop(['target'], axis=1).loc[train_indices, :],
                             label=train.loc[train_indices, 'target'])
    val_data = lgb.Dataset(train.drop(['target'], axis=1).loc[val_indices, :], label=train.loc[val_indices, 'target'])

    params = {
            'objective': 'binary',
            'boosting': 'dart',
            'learning_rate': 0.3 ,
            'verbose': 0,
            'num_leaves': 108,
            'bagging_fraction': 0.95,
            'bagging_freq': 1,
            'bagging_seed': 1,
            'feature_fraction': 0.9,
            'feature_fraction_seed': 1,
            'max_bin': 256,
            # 'max_depth': 10,
            'max_depth': 12,
            'num_rounds': 200,
            'metric' : 'auc'
        }

    bst = lgb.train(params, train_data, 200, valid_sets=[val_data],verbose_eval=5)
    y_submission=bst.predict(train.drop(['target'], axis=1).loc[val_indices, :])  #每折的预测
    dataset_blend_train[val_indices, 1] = y_submission
    dataset_blend_test_j[:, i] = bst.predict(test.drop(['id'], axis=1))
    del bst
dataset_blend_test[:,1] = dataset_blend_test_j.mean(1)

print("Blending.")
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()

#模型预测结果(几个模型就有几列），真实结果（1列）
clf.fit(dataset_blend_train, y_train)

#模型对test的预测结果（几个模型，test就有几列）
y_submission = clf.predict_proba(dataset_blend_test)[:, 1]
print(y_submission)
print(len(y_submission))
submission = pd.read_csv('sample_submission.csv')
submission.target = y_submission
submission.to_csv('sub_stacking_2000_1217.csv', index=False)
# del train, test; gc.collect();
#
# d_train_final = lgb.Dataset(X_train, y_train)
# watchlist_final = lgb.Dataset(X_train, y_train)
# print('Processed data...')
#
# params = {
#         'objective': 'binary',
#         'boosting': 'gbdt',
#         'learning_rate': 0.3 ,
#         'verbose': 0,
#         'num_leaves': 108,
#         'bagging_fraction': 0.95,
#         'bagging_freq': 1,
#         'bagging_seed': 1,
#         'feature_fraction': 0.9,
#         'feature_fraction_seed': 1,
#         'max_bin': 256,
#         #'max_depth': 10,
#         'max_depth': 12,
#         'num_rounds': 200,
#         'metric' : 'auc'
#     }
#
# model_f1 = lgb.train(params, train_set=d_train_final,  valid_sets=watchlist_final, verbose_eval=5)
# import matplotlib.pyplot as plt
# predictors  = [i for i in X_train .columns]
# feat_imp = pd.Series(model_f1.feature_importance(importance_type='split',iteration=-1), predictors).sort_values(ascending=False)
# feat_imp.to_csv('feature importances_1213_xiawu.csv')
# # feat_imp.plot(kind='bar', title='Feature Importances')
# # plt.ylabel('Feature Importance Score')
# # plt.show()
#
# params = {
#         'objective': 'binary',
#         'boosting': 'dart',
#         'learning_rate': 0.3 ,
#         'verbose': 0,
#         'num_leaves': 108,
#         'bagging_fraction': 0.95,
#         'bagging_freq': 1,
#         'bagging_seed': 1,
#         'feature_fraction': 0.9,
#         'feature_fraction_seed': 1,
#         'max_bin': 256,
#         # 'max_depth': 10,
#         'max_depth': 12,
#         'num_rounds': 200,
#         'metric' : 'auc'
#     }
#
# model_f2 = lgb.train(params, train_set=d_train_final,  valid_sets=watchlist_final, verbose_eval=5)
#
# feat_imp = pd.Series(model_f2.feature_importance(importance_type='split',iteration=-1), predictors).sort_values(ascending=False)
# feat_imp.to_csv('feature importances_1213.csv')
# # feat_imp.plot(kind='bar', title='Feature Importances')
# # plt.ylabel('Feature Importance Score')
# # plt.show()
#
# print('Making predictions')
# p_test_1 = model_f1.predict(X_test)
# p_test_2 = model_f2.predict(X_test)
# p_test_avg = np.mean([p_test_1, p_test_2], axis = 0)
#
#
# print('Done making predictions')
#
# print ('Saving predictions Model model of gbdt')
#
# subm = pd.DataFrame()
# subm['id'] = ids
# subm['target'] = p_test_avg
# subm.to_csv('submission_1213.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')
#
# print('Done!')
