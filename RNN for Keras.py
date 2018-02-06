# coding=utf-8
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import os
import gc
import time
start_time = time.time()

# import matplotlib.pyplot as plt
# import seaborn as sns
# import scipy
# from sklearn.model_selection import train_test_split, cross_val_score


NUM_BRANDS = 4000
NUM_CATEGORIES = 1000
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 40000

df_train = train = pd.read_csv(r'D:\Alibaba Game\MercariPriceSuggestion\train.tsv', sep='\t')
df_test = test = pd.read_csv(r'D:\Alibaba Game\MercariPriceSuggestion\test.tsv', sep='\t')
train['target'] = np.log1p(train['price']) # log1p 的作用等同于log(x+1)
test_len = test.shape[0]


# ############# 做模拟测试来缩放数据集 ##################
def simulate_test(test):
    # 这段代码很奇怪，如果test的长度小于800000的话，就在test里面随机抽取数来组成2800000的序列
    # 然后再将这两部分的数据拼接到一起，起到了一种缩放数据集的作用，但是效果不详
    # 查阅了一些资料，最后在kaggle一大神那里得知，这个是在当时kaggle比赛中第二步的加时规则没有
    # 出台的时候常用的手法，可能是更好地利用现有数据的信息量，给他缩放了一下，在此次比赛中并没有什么实际意义，可以丢掉
    print('5 folds scaling the test_df')
    if test.shape[0] < 800000:
        indices = np.random.choice(test.index.values, 2800000)
        test_ = pd.concat([test, test.iloc[indices]], axis=0)
        return test_
    else:
        return test


print(test)
print("this is the new test")
test = simulate_test(test)
print(test)
print('new shape ', test.shape)
print('[{}] Finished scaling test set...'.format(time.time() - start_time))


# #############HANDLE MISSING VALUES###################################
def handle_missing(dataset):
    print("Handling missing values...")
    dataset.category_name.fillna(value="missing", inplace=True)
    dataset.brand_name.fillna(value="missing", inplace=True)
    dataset.item_description.fillna(value="missing", inplace=True)
    return (dataset)
train = handle_missing(train)
test = handle_missing(test)
print('[{}] Finished handling missing data...'.format(time.time() - start_time))

# PROCESS CATEGORICAL DATA


print("Handling categorical variables...")
le = LabelEncoder()
le.fit(np.hstack([train.category_name, test.category_name]))
# hstack的作用就是将一串数组用堆栈压成一条数组，如果想恢复原来的数组的话，就需要hsplit
train['category'] = le.transform(train.category_name)
# labelencoder 的transform是将标签标准化
test['category'] = le.transform(test.category_name)
le.fit(np.hstack([train.brand_name, test.brand_name]))
# labelencoder的 fit是对目标进行标签编码
train['brand'] = le.transform(train.brand_name)
test['brand'] = le.transform(test.brand_name)
del le, train['brand_name'], test['brand_name']

print('[{}] Finished PROCESSING CATEGORICAL DATA...'.format(time.time() - start_time))
train.head(3)

# PROCESS TEXT: RAW
print("Text to seq process...")
print("   Fitting tokenizer...")
from keras.preprocessing.text import Tokenizer

raw_text = np.hstack([train.category_name.str.lower(),
                      train.item_description.str.lower(),
                      train.name.str.lower()])

tok_raw = Tokenizer()
# Tokenizer是一个用于向量化文本，或将文本转换为序列（即单词在字典中的下标构成的列表，从1算起）的类。
tok_raw.fit_on_texts(raw_text)
print("   Transforming text to seq...")
train["seq_category_name"] = tok_raw.texts_to_sequences(train.category_name.str.lower())
test["seq_category_name"] = tok_raw.texts_to_sequences(test.category_name.str.lower())
train["seq_item_description"] = tok_raw.texts_to_sequences(train.item_description.str.lower())
test["seq_item_description"] = tok_raw.texts_to_sequences(test.item_description.str.lower())
train["seq_name"] = tok_raw.texts_to_sequences(train.name.str.lower())
test["seq_name"] = tok_raw.texts_to_sequences(test.name.str.lower())
train.head(3)

print('[{}] Finished PROCESSING TEXT DATA...'.format(time.time() - start_time))

# EXTRACT DEVELOPTMENT TEST
from sklearn.model_selection import train_test_split

dtrain, dvalid = train_test_split(train, random_state=666, train_size=0.99)

# EMBEDDINGS MAX VALUE
# Base on the histograms, we select the next lengths
MAX_NAME_SEQ = 20  # 17
MAX_ITEM_DESC_SEQ = 60  # 269
MAX_CATEGORY_NAME_SEQ = 20  # 8
MAX_TEXT = np.max([np.max(train.seq_name.max()),
                   np.max(test.seq_name.max()),
                   np.max(train.seq_category_name.max()),
                   np.max(test.seq_category_name.max()),
                   np.max(train.seq_item_description.max()),
                   np.max(test.seq_item_description.max())]) + 2
MAX_CATEGORY = np.max([train.category.max(), test.category.max()]) + 1
MAX_BRAND = np.max([train.brand.max(), test.brand.max()]) + 1
MAX_CONDITION = np.max([train.item_condition_id.max(),
                        test.item_condition_id.max()]) + 1

print('[{}] Finished EMBEDDINGS MAX VALUE...'.format(time.time() - start_time))

# KERAS DATA DEFINITION
from keras.preprocessing.sequence import pad_sequences


def get_keras_data(dataset):
    X = {
        'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ),
        'item_desc': pad_sequences(dataset.seq_item_description
                                     , maxlen=MAX_ITEM_DESC_SEQ),
        # pad_sequence 的作用就是将序列都填补到一样长。默认填补到最长序列的长度。
        # 如果有了最大长度的限制的话，那么大于这一长度的序列都会被截断，从最开始(默认)或者结束的地方截
        # 支持预填补和后填充
        'brand': np.array(dataset.brand),
        'category': np.array(dataset.category),
        'category_name': pad_sequences(dataset.seq_category_name
                                         , maxlen=MAX_CATEGORY_NAME_SEQ),
        'item_condition': np.array(dataset.item_condition_id),
        'num_vars': np.array(dataset[["shipping"]])
    }
    return X


X_train = get_keras_data(dtrain)
X_valid = get_keras_data(dvalid)
X_test = get_keras_data(test)
print(dtrain)
print("?///////////////////////////")
print(X_train)
print('[{}] Finished DATA PREPARARTION...'.format(time.time() - start_time))

# KERAS MODEL DEFINITION
from keras.layers import Input, Dropout, Dense, BatchNormalization, \
    Activation, concatenate, GRU, Embedding, Flatten
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping  # , TensorBoard
from keras import backend as K
from keras import optimizers
from keras import initializers


def rmsle(y, y_pred):
    import math
    assert len(y) == len(y_pred)
    to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 \
              for i, pred in enumerate(y_pred)]
    return (sum(to_sum) * (1.0 / len(y))) ** 0.5
# 误差计算公式


dr = 0.25


def get_model():
    # params
    dr_r = dr

    # Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand = Input(shape=[1], name="brand")
    category = Input(shape=[1], name="category")
    category_name = Input(shape=[X_train["category_name"].shape[1]],
                          name="category_name")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")

    # Embeddings layers
    emb_size = 60

    emb_name = Embedding(MAX_TEXT, emb_size // 3)(name)
    emb_item_desc = Embedding(MAX_TEXT, emb_size)(item_desc)
    emb_category_name = Embedding(MAX_TEXT, emb_size // 3)(category_name)
    emb_brand = Embedding(MAX_BRAND, 10)(brand)
    emb_category = Embedding(MAX_CATEGORY, 10)(category)
    emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)
    # embedding的意义在于将原来的词变成向量

    rnn_layer1 = GRU(16)(emb_item_desc)
    rnn_layer2 = GRU(8)(emb_category_name)
    rnn_layer3 = GRU(8)(emb_name)

    # main layer
    main_l = concatenate([
        Flatten()(emb_brand),
        Flatten()(emb_category),
        Flatten()(emb_item_condition),
        # Flatten层表达的意思就是我们将一个矩阵拍扁成一个向量，也就是将每一行连接起来，
        # 构建一个长向量
        rnn_layer1,
        rnn_layer2,
        rnn_layer3,
        num_vars,
    ])     # concatenate 函数是要构建一个连锁的神经网络层，
    main_l = Dropout(0.2)(Dense(512, activation='relu')(main_l))
    main_l = Dropout(0.2)(Dense(64, activation='relu')(main_l))


    # output
    output = Dense(1, activation="linear")(main_l)

    # model
    model = Model([name, item_desc, brand
                      , category, category_name
                      , item_condition, num_vars], output)
    # optimizer = optimizers.RMSprop()
    optimizer = optimizers.Adam()
    model.compile(loss="mse",
                  optimizer=optimizer)
    return model


def eval_model(model):
    val_preds = model.predict(X_valid)
    val_preds = np.expm1(val_preds)   # expm1 的作用就是exp(x) - 1

    y_true = np.array(dvalid.price.values)
    y_pred = val_preds[:, 0]
    v_rmsle = rmsle(y_true, y_pred)
    # 如果预测的值范围很大，RMSE就会被一些很大的值主导。这样即使很多小的值预测准了，但是一个非常大的值预测不准，
    # RMSE 就会变得很大。所以RMSE 对于较大的偏差具有非常好的约束性
    print(" RMSLE error on dev test: " + str(v_rmsle))
    return v_rmsle


exp_decay = lambda init, fin, steps: (init / fin) ** (1 / (steps - 1)) - 1

print('[{}] Finished DEFINEING MODEL...'.format(time.time() - start_time))

gc.collect()
# FITTING THE MODEL
epochs = 2
BATCH_SIZE = 512 * 3
steps = int(len(X_train['name']) / BATCH_SIZE) * epochs
lr_init, lr_fin = 0.009, 0.006
lr_decay = exp_decay(lr_init, lr_fin, steps)
# log_subdir = '_'.join(['ep', str(epochs),
#                        'bs', str(BATCH_SIZE),
#                        'lrI', str(lr_init),
#                        'lrF', str(lr_fin),
#                        'dr', str(dr)])  # 在迭代中将这几个字符串连接起来
# print("连接起来的样子为"+log_subdir)

model = get_model()
K.set_value(model.optimizer.lr, lr_init)
K.set_value(model.optimizer.decay, lr_decay)
history = model.fit(X_train, dtrain.target
                    , epochs=epochs
                    , batch_size=BATCH_SIZE
                    , validation_split=0.2
                    # , callbacks=[TensorBoard('./logs/'+log_subdir)]
                    , verbose=1
                    )
print('[{}] Finished FITTING MODEL...'.format(time.time() - start_time))
# EVLUEATE THE MODEL ON DEV TEST
v_rmsle = eval_model(model)
print('[{}] Finished predicting valid set...'.format(time.time() - start_time))

# CREATE PREDICTIONS
preds = model.predict(X_test, batch_size=BATCH_SIZE)
preds = np.expm1(preds)
print('[{}] Finished predicting test set...'.format(time.time() - start_time))
submission = test[["test_id"]][:test_len]
submission["price"] = preds[:test_len]
# submission["price"] = preds[:test_len] * 0.78
print('[{}] Finished predicting test set...'.format(time.time() - start_time))

del train
del test
gc.collect()  # 垃圾回收
print("the gc collection is", gc.collect())
nrow_train = df_train.shape[0]
y = np.log1p(df_train["price"])
merge = pd.concat([df_train, df_test])

del df_train
del df_test
# gc.collect()


def handle_missing_inplace(dataset):
    dataset['category_name'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)


handle_missing_inplace(merge)
print('[{}] Finished to handle missing'.format(time.time() - start_time))


def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category = dataset['category_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['category_name'].isin(pop_category), 'category_name'] = 'missing'
cutting(merge)
print('[{}] Finished to cut'.format(time.time() - start_time))


def to_categorical(dataset):
    dataset['category_name'] = dataset['category_name'].astype('category')
    dataset['brand_name'] = dataset['brand_name'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')
to_categorical(merge)
print('[{}] Finished to convert categorical'.format(time.time() - start_time))


# ###################对几个重要的特征进行编码转换处理 ####################################################################
cv = CountVectorizer(min_df=NAME_MIN_DF)
X_name = cv.fit_transform(merge['name'])   # X_name被列表化，特征向量转化成标准的Python字典对象的一个列表
print('[{}] Finished count vectorize `name`'.format(time.time() - start_time))

cv = CountVectorizer()
X_category = cv.fit_transform(merge['category_name'])   # X_category 被列表化，特征向量转化成标准的Python字典对象的一个列表
print('[{}] Finished count vectorize `category_name`'.format(time.time() - start_time))

tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
                     ngram_range=(1, 3),
                     stop_words='english')
X_description = tv.fit_tran2sform(merge['item_description'])   # X_description 使用TF-IDF转换
print('[{}] Finished TFIDF vectorize `item_description`'.format(time.time() - start_time))

lb = LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(merge['brand_name'])     # X_brand进行标签编码
print('[{}] Finished label binarize `brand_name`'.format(time.time() - start_time))

X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                      sparse=True).values)  # get_dummies进行one_hot编码
print('[{}] Finished to get dummies on `item_condition_id` and `shipping`'.format(time.time() - start_time))

sparse_merge = hstack((X_dummies, X_description, X_brand, X_category, X_name)).tocsr()
print('[{}] Finished to create sparse merge'.format(time.time() - start_time))  # 对这五个特征进行整合成单个向量
X = sparse_merge[:nrow_train]
X_test = sparse_merge[nrow_train:]    # 将单个向量进行前后部分拆分


alpha_Ridge = [2.5, 3.5]
rate_Ridge = [0.07, 0.13]
for i in range(0, 1):
    model = Ridge(solver="sag", fit_intercept=True, alpha=alpha_Ridge[i], random_state=666)
    model.fit(X, y)
    print('[{}] Finished to train ridge '.format(time.time() - start_time), i)
    predsRR = model.predict(X=X_test)
    print('[{}] Finished to predict ridge '.format(time.time() - start_time), i)
    predsRR = np.expm1(predsRR)
    predsRR = predsRR * rate_Ridge[i]
    submission["price"] += predsRR
# Ridge（岭回归）是一种最小二乘法修正过的方法。sag为Stochastic Average Gradient descent，fit_intercept为是否计算截距的开关，
# alpha为正则化的强度
# 在线性方程无解的时候，我们可以将线性方程转化成矩阵形式，这样将目标投影到对应解空间，则会求到一个最近似的解
# 而我们将这个过程重新展开会发现，其实是最小二乘来最小化误差的过程。而在矩阵A并非满秩的时候，我们也无法求解这个最小二乘，
# 岭回归就给其加入l2正则化，添加一点扰动，使其满秩。正则化还有惩罚估计参数beta的作用，防止beta过大
# 由于最小二乘是无偏估计，而加入了正则化的岭回归是有偏估计，所以偏差有了，但是提高了稳定性


l1_ratio_seq = [0.05, 0.15, 0.25, 0.5, 1]
for i in range(0, 4):
    model = SGDRegressor(penalty='elasticnet', l1_ratio=l1_ratio_seq[i])
    model.fit(X, y).sparsify()
    print('[{}] Train sgd completed'.format(time.time() - start_time))
    predsR2 = model.predict(X=X_test)
    predsR2 = np.expm1(predsR2)
    predsR2 = predsR2 * 0.005
    submission["price"] += predsR2
print('[{}] Predict sgd completed.'.format(time.time() - start_time))
submission.to_csv("submission_rnn_ridge_sgdr.csv", index=False)