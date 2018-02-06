import numpy as np
from datetime import datetime

import pandas as pd
import lightgbm as lgb
import gc
import csv
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten, Activation
# from keras.layers import Bidirectional
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
import math
# set seed
start_real = datetime.datetime.now()
np.random.seed(123)


def rmsle(Y, Y_pred):
    assert Y.shape == Y_pred.shape
    return np.sqrt(np.mean(np.square(Y_pred - Y)))
# 误差计算公式


train_df = pd.read_table(r'D:\MercariPriceSuggestion\train.tsv')
test_df = pd.read_table(r'D:\MercariPriceSuggestion\test.tsv')

# remove low prices and high prices
train_df = train_df.drop(train_df[(train_df.price < 3.0)].index)
# print(train_df.shape)          # (1481658, 8)

train_df = train_df.drop(train_df[(train_df.price > 2000)].index)
# print(train_df.shape)            # (1481658,8)


# get name and description lengths
def wordCount(text):
    try:
        if text == 'No description yet':
            return 0
        else:
            text = text.lower()
            words = [w for w in text.split(" ")]
            return len(words)
    except:
        return 0


train_df['desc_len'] = train_df['item_description'].apply(lambda x: wordCount(x))
test_df['desc_len'] = test_df['item_description'].apply(lambda x: wordCount(x))
train_df['name_len'] = train_df['name'].apply(lambda x: wordCount(x))
test_df['name_len'] = test_df['name'].apply(lambda x: wordCount(x))
train_df.head()


# split category name into 3 parts
def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")
train_df['subcat_0'], train_df['subcat_1'], train_df['subcat_2'] = \
zip(*train_df['category_name'].apply(lambda x: split_cat(x)))
test_df['subcat_0'], test_df['subcat_1'], test_df['subcat_2'] = \
zip(*test_df['category_name'].apply(lambda x: split_cat(x)))


# attempt to find missing brand names
# train_df['name'] = train_df.name.str.lower()
# train_df['brand_name'] = train_df.brand_name.str.lower()
# test_df['name'] = test_df.name.str.lower()
# test_df['brand_name'] = test_df.brand_name.str.lower()
full_set = pd.concat([train_df,test_df])
all_brands = set(full_set['brand_name'].values)
train_df.brand_name.fillna(value="missing", inplace=True)
test_df.brand_name.fillna(value="missing", inplace=True)

# get to finding!
premissing = len(train_df.loc[train_df['brand_name'] == 'missing'])
def brandfinder(line):
    brand = line[0]
    name = line[1]
    namesplit = name.split(' ')
    if brand == 'missing':
        for x in namesplit:
            if x in all_brands:
                return name
    if name in all_brands:
        return name
    return brand


train_df['brand_name'] = train_df[['brand_name','name']].apply(brandfinder, axis = 1)
test_df['brand_name'] = test_df[['brand_name','name']].apply(brandfinder, axis = 1)
found = premissing-len(train_df.loc[train_df['brand_name'] == 'missing'])            #137342


# Scale target variable to log.
train_df["target"] = np.log1p(train_df.price)

# Split training examples into train/dev examples.
train_df, dev_df = train_test_split(train_df, random_state=123, train_size=0.8)
# print("the train_df set is: ", train_df)
# Calculate number of train/dev/test examples.
n_trains = train_df.shape[0]
n_devs = dev_df.shape[0]
n_tests = test_df.shape[0]
print("Training on", n_trains, "examples")
print("Validating on", n_devs, "examples")
print("Testing on", n_tests, "examples")


#---------------------RNN-----------------------------#

# Concatenate train - dev - test data for easy to handle
full_df = pd.concat([train_df, dev_df, test_df])

# Filling missing values
def fill_missing_values(df):
    df.category_name.fillna(value="missing", inplace=True)
    df.brand_name.fillna(value="missing", inplace=True)
    df.item_description.fillna(value="missing", inplace=True)
    df.item_description.replace('No description yet',"missing", inplace=True)
    return df

print("Filling missing data...")
full_df = fill_missing_values(full_df)
# print(full_df.category_name[1])


# Processing categorical data
print("Processing categorical data...")
le = LabelEncoder()
# full_df.category = full_df.category_name
le.fit(full_df.category_name)
full_df['category'] = le.transform(full_df.category_name)

le.fit(full_df.brand_name)
full_df.brand_name = le.transform(full_df.brand_name)

le.fit(full_df.subcat_0)
full_df.subcat_0 = le.transform(full_df.subcat_0)

le.fit(full_df.subcat_1)
full_df.subcat_1 = le.transform(full_df.subcat_1)

le.fit(full_df.subcat_2)
full_df.subcat_2 = le.transform(full_df.subcat_2)

del le

#Process text data

print("Transforming text data to sequences...")
raw_text = np.hstack([full_df.item_description.str.lower(), full_df.name.str.lower(), full_df.category_name.str.lower()])

print("   Fitting tokenizer...")
tok_raw = Tokenizer()
tok_raw.fit_on_texts(raw_text)

print("   Transforming text to sequences...")
full_df['seq_item_description'] = tok_raw.texts_to_sequences(full_df.item_description.str.lower())
full_df['seq_name'] = tok_raw.texts_to_sequences(full_df.name.str.lower())
# full_df['seq_category'] = tok_raw.texts_to_sequences(full_df.category_name.str.lower())

del tok_raw

print(full_df['seq_name'][:5])

#Define constants to use when define RNN model
MAX_NAME_SEQ = 10 #17
MAX_ITEM_DESC_SEQ = 75 #269
MAX_CATEGORY_SEQ = 8 #8
MAX_TEXT = np.max([
    np.max(full_df.seq_name.max()),
    np.max(full_df.seq_item_description.max()),
#     np.max(full_df.seq_category.max()),
]) + 100
MAX_CATEGORY = np.max(full_df.category.max()) + 1
MAX_BRAND = np.max(full_df.brand_name.max()) + 1
MAX_CONDITION = np.max(full_df.item_condition_id.max()) + 1
MAX_DESC_LEN = np.max(full_df.desc_len.max()) + 1
MAX_NAME_LEN = np.max(full_df.name_len.max()) + 1
MAX_SUBCAT_0 = np.max(full_df.subcat_0.max()) + 1
MAX_SUBCAT_1 = np.max(full_df.subcat_1.max()) + 1
MAX_SUBCAT_2 = np.max(full_df.subcat_2.max()) + 1
print("the maxcatogory is : ",MAX_CATEGORY)
#Get data for RNN model
def get_rnn_data(dataset):
    X = {
        'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ),
        'item_desc': pad_sequences(dataset.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ),
        'brand_name': np.array(dataset.brand_name),
        'category': np.array(dataset.category),
#         'category_name': pad_sequences(dataset.seq_category, maxlen=MAX_CATEGORY_SEQ),
        'item_condition': np.array(dataset.item_condition_id),
        'num_vars': np.array(dataset[["shipping"]]),
        'desc_len': np.array(dataset[["desc_len"]]),
        'name_len': np.array(dataset[["name_len"]]),
        'subcat_0': np.array(dataset.subcat_0),
        'subcat_1': np.array(dataset.subcat_1),
        'subcat_2': np.array(dataset.subcat_2),
    }
    return X

train = full_df[:n_trains]
dev = full_df[n_trains:n_trains+n_devs]
test = full_df[n_trains+n_devs:]
# print("the train's price is: ",dev)
X_train = get_rnn_data(train)
Y_train = train.target.values.reshape(-1, 1)

X_dev = get_rnn_data(dev)
Y_dev = dev.target.values.reshape(-1, 1)

X_test = get_rnn_data(test)

def root_mean_squared_logarithmic_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1)+0.0000001)
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)+0.0000001)

#Define RNN model

# set seed again in case testing models adjustments by looping next 2 blocks
np.random.seed(123)


def new_rnn_model(lr=0.001, decay=0.0):
    # Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand_name = Input(shape=[1], name="brand_name")
    #     category = Input(shape=[1], name="category")
    #     category_name = Input(shape=[X_train["category_name"].shape[1]], name="category_name")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")
    desc_len = Input(shape=[1], name="desc_len")
    name_len = Input(shape=[1], name="name_len")
    subcat_0 = Input(shape=[1], name="subcat_0")
    subcat_1 = Input(shape=[1], name="subcat_1")
    subcat_2 = Input(shape=[1], name="subcat_2")

    # Embeddings layers (adjust outputs to help model)
    emb_name = Embedding(MAX_TEXT, 20)(name)
    emb_item_desc = Embedding(MAX_TEXT, 60)(item_desc)
    emb_brand_name = Embedding(MAX_BRAND, 10)(brand_name)
    #     emb_category_name = Embedding(MAX_TEXT, 20)(category_name)
    #     emb_category = Embedding(MAX_CATEGORY, 10)(category)
    emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)
    emb_desc_len = Embedding(MAX_DESC_LEN, 5)(desc_len)
    emb_name_len = Embedding(MAX_NAME_LEN, 5)(name_len)
    emb_subcat_0 = Embedding(MAX_SUBCAT_0, 10)(subcat_0)
    emb_subcat_1 = Embedding(MAX_SUBCAT_1, 10)(subcat_1)
    emb_subcat_2 = Embedding(MAX_SUBCAT_2, 10)(subcat_2)

    # rnn layers (GRUs are faster than LSTMs and speed is important here)
    rnn_layer1 = GRU(16)(emb_item_desc)
    rnn_layer2 = GRU(8)(emb_name)
    #     rnn_layer3 = GRU(8) (emb_category_name)

    # main layers
    main_l = concatenate([
        Flatten()(emb_brand_name)
        #         , Flatten() (emb_category)
        , Flatten()(emb_item_condition)
        , Flatten()(emb_desc_len)
        , Flatten()(emb_name_len)
        , Flatten()(emb_subcat_0)
        , Flatten()(emb_subcat_1)
        , Flatten()(emb_subcat_2)
        , rnn_layer1
        , rnn_layer2
        #         , rnn_layer3
        , num_vars
    ])
    # (incressing the nodes or adding layers does not effect the time quite as much as the rnn layers)
    main_l = Dropout(0.1)(Dense(512, kernel_initializer='normal', activation='relu')(main_l))
    main_l = Dropout(0.1)(Dense(256, kernel_initializer='normal', activation='relu')(main_l))
    main_l = Dropout(0.1)(Dense(128, kernel_initializer='normal', activation='relu')(main_l))
    main_l = Dropout(0.1)(Dense(64, kernel_initializer='normal', activation='relu')(main_l))

    # the output layer.
    output = Dense(1, activation="linear")(main_l)

    model = Model([name, item_desc, brand_name, item_condition,
                   num_vars, desc_len, name_len, subcat_0, subcat_1, subcat_2], output)

    optimizer = Adam(lr=lr, decay=decay)
    # (mean squared error loss function works as well as custom functions)
    model.compile(loss='mse', optimizer=optimizer)

    return model


model = new_rnn_model()
model.summary()
del model

#Fit RNN model to train data

# Set hyper parameters for the model.
BATCH_SIZE = 512 * 3
epochs = 2

# Calculate learning rate decay.
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(X_train['name']) / BATCH_SIZE) * epochs
lr_init, lr_fin = 0.005, 0.001
lr_decay = exp_decay(lr_init, lr_fin, steps)

# Create model and fit it with training dataset.
rnn_model = new_rnn_model(lr=lr_init, decay=lr_decay)
rnn_model.fit(
        X_train, Y_train, epochs=epochs, batch_size=BATCH_SIZE,
        validation_data=(X_dev, Y_dev), verbose=2,
)

# Evaluate RNN model on dev data
print("Evaluating the model on validation data...")
Y_dev_preds_rnn = rnn_model.predict(X_dev, batch_size=BATCH_SIZE)
# print(Y_dev_preds_rnn)
print(" RMSLE error:", rmsle(Y_dev, Y_dev_preds_rnn))
submissionForRNN = pd.DataFrame({
        "test_id": test_df.test_id,
        "price": Y_dev_preds_rnn,
})

submissionForRNN = submissionForRNN[['test_id', 'price']]
submissionForRNN.to_csv("./submissionForRNN.csv", index=False)

# #Make prediction for test data
# rnn_preds = rnn_model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
# rnn_preds = np.expm1(rnn_preds)
# submissionForRNN = pd.DataFrame({
#         "test_id": test_df.test_id,
#         "price": rnn_preds.reshape(-1),
# })
# submissionForRNN = submissionForRNN[['test_id', 'price']]


df_train = pd.read_csv(r'D:\MercariPriceSuggestion\train.tsv', sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE)
df_test = pd.read_csv(r'D:\MercariPriceSuggestion\test.tsv', sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE)


print('Load data complete\nShape train: {}\nShape test: {}'.format(df_train.shape, df_test.shape))
median_price = df_train['price'].median()
mean_price = df_train['price'].mean()
exclude_cols = ['name', 'category_name', 'brand_name', 'item_description', 'category_1', 'category_2', 'category_3']


def category_detail(x, i):
    try:
        x = x.split('/')[i]
        return x
    except:
        return ''


def category_detail_1(x):
    return category_detail(x, 0)


def category_detail_2(x):
    return category_detail(x, 1)


def category_detail_3(x):
    return category_detail(x, 2)


def price_features_groupby(df_train, col, type):
    if (type == 'median'):
        price_dict = df_train.groupby(col)['price'].median().to_dict()
    elif (type == 'mean'):
        price_dict = df_train.groupby(col)['price'].mean().to_dict()
    elif (type == 'max'):
        price_dict = df_train.groupby(col)['price'].max().to_dict()
    else:
        price_dict = df_train.groupby(col)['price'].min().to_dict()
    tmp = pd.DataFrame({
        col: list(price_dict.keys()),
        '{}_{}_price'.format(col, type): list(price_dict.values())})
    return tmp


def price_features(df_train, df_test, col, type):
    na_value = -1

    if (type == 'median'):
        na_value = median_price
    elif (type == 'mean'):
        na_value = mean_price

    tmp = price_features_groupby(df_train, col, type)
    df_train = pd.merge(df_train, tmp, how='left', on=col)
    df_train['{}_{}_price'.format(col, type)].fillna(na_value, inplace=True)
    df_train['{}_{}_price'.format(col, type)] = df_train['{}_{}_price'.format(col, type)].astype(np.int16)

    df_test = pd.merge(df_test, tmp, how='left', on=col)
    df_test['{}_{}_price'.format(col, type)].fillna(na_value, inplace=True)
    df_test['{}_{}_price'.format(col, type)] = df_test['{}_{}_price'.format(col, type)].astype(np.int16)

    return df_train, df_test


def category_features(df_train, df_test, col):
    cate_train = set(df_train[col].unique())
    cate_test = set(df_test[col].unique())
    cate_all = cate_train.union(cate_test)
    print('category {} in train have {} unique values'.format(col, len(cate_train)))
    print('category {} in test have {} unique values'.format(col, len(cate_test)))
    print('category {} in train ∪ test have {} unique values'.format(col, len(cate_all)))
    print()
    tmp = pd.DataFrame({
        col: list(cate_all),
        '{}_cat'.format(col): [i + 1 for i in range(len(cate_all))]})
    df_train = pd.merge(df_train, tmp, how='left', on=col)
    df_train['{}_cat'.format(col)].fillna(-1, inplace=True)
    df_test = pd.merge(df_test, tmp, how='left', on=col)
    df_test['{}_cat'.format(col)].fillna(-1, inplace=True)
    return df_train, df_test


df_train['category_1'] = df_train['category_name'].apply(category_detail_1)
df_train['category_2'] = df_train['category_name'].apply(category_detail_2)
df_train['category_3'] = df_train['category_name'].apply(category_detail_3)

df_test['category_1'] = df_test['category_name'].apply(category_detail_1)
df_test['category_2'] = df_test['category_name'].apply(category_detail_2)
df_test['category_3'] = df_test['category_name'].apply(category_detail_3)

for col in ['category_name', 'brand_name', 'category_1', 'category_2', 'category_3']:
    df_train, df_test = category_features(df_train, df_test, col)

for col in ['category_name', 'brand_name', 'category_1', 'category_2', 'category_3', 'item_condition_id']:
    for type in ['median', 'mean', 'max', 'min']:
        df_train, df_test = price_features(df_train, df_test, col, type)

# start https://www.kaggle.com/golubev/naive-xgboost-v2
c_texts = ['name', 'item_description']


def count_words(key):
    return len(str(key).split())


def count_numbers(key):
    return sum(c.isalpha() for c in str(key))


def count_upper(key):
    return sum(c.isupper() for c in str(key))


for c in c_texts:
    df_train[c + '_c_words'] = df_train[c].apply(count_words)
    df_train[c + '_c_upper'] = df_train[c].apply(count_upper)
    df_train[c + '_c_numbers'] = df_train[c].apply(count_numbers)
    df_train[c + '_len'] = df_train[c].str.len()
    df_train[c + '_mean_len_words'] = df_train[c + '_len'] / df_train[c + '_c_words']
    df_train[c + '_mean_upper'] = df_train[c + '_len'] / df_train[c + '_c_upper']
    df_train[c + '_mean_numbers'] = df_train[c + '_len'] / df_train[c + '_c_numbers']

for c in c_texts:
    df_test[c + '_c_words'] = df_test[c].apply(count_words)
    df_test[c + '_c_upper'] = df_test[c].apply(count_upper)
    df_test[c + '_c_numbers'] = df_test[c].apply(count_numbers)
    df_test[c + '_len'] = df_test[c].str.len()
    df_test[c + '_mean_len_words'] = df_test[c + '_len'] / df_test[c + '_c_words']
    df_test[c + '_mean_upper'] = df_test[c + '_len'] / df_test[c + '_c_upper']
    df_test[c + '_mean_numbers'] = df_test[c + '_len'] / df_test[c + '_c_numbers']
# end https://www.kaggle.com/golubev/naive-xgboost-v2


print('After adding features\nShape train: {}\nShape test: {}'.format(df_train.shape, df_test.shape))

gc.collect()
# from https://www.kaggle.com/shikhar1/base-random-forest-lb-532
df_train['price'] = df_train['price'].apply(lambda x: np.log(x + 1))
target_train = df_train['price'].values
train = np.array(df_train['train_id'])
df_train = df_train.drop(['train_id', 'price'] + exclude_cols, axis=1)

cat_features = []
for i, c in enumerate(df_train.columns):
    if ('_cat' in c):
        cat_features.append(c)

params = {
    'learning_rate': 0.25,
    'application': 'regression',
    'max_depth': 5,
    'num_leaves': 256,
    'verbosity': -1,
    'metric': 'RMSE'
}
moedels = []

train_X, valid_X, train_y, valid_y = train_test_split(df_train, target_train, test_size=0.2, random_state=666)
d_train = lgb.Dataset(train_X, label=train_y)
d_valid = lgb.Dataset(valid_X, label=valid_y)
watchlist = [d_train, d_valid]

model = lgb.train(params, train_set=d_train, num_boost_round=240, valid_sets=watchlist, \
                      early_stopping_rounds=20, verbose_eval=10, categorical_feature=cat_features)
moedels.append(model)
ax = lgb.plot_importance(model)
plt.tight_layout()
plt.savefig('feature_importance_{}.png'.format(i))

pred_For_LGB = model.predict(valid_X)
submissionForLGB = pd.DataFrame({
        "test_id": test_df.test_id,
        "price": pred_For_LGB,
})

submissionForLGB = submissionForLGB[['test_id', 'price']]
submissionForLGB.to_csv("./submissionForLGB.csv", index=False)

# for i in range(0,100):
#     predForValid = pred_For_LGB*i + Y_dev_preds_rnn*(100-i)
#     print("///////////////////////////////////////////////////")
#     print("i = ",i)
#     print("the rmsle of model is: ", rmsle(valid_y, predForValid))
# del target_train, train
# gc.collect()

# train_end_datetime = datetime.datetime.now()
# print('Training end at {}'.format(train_end_datetime.strftime("%Y-%m-%d %H:%M:%S")))
# print('Total training time: {} seconds'.format((train_end_datetime - start_real).total_seconds()))
# preds = None
# for i, model in enumerate(moedels):
#     pred = model.predict(df_test.drop(['test_id'] + exclude_cols, axis=1))
#     pred = pd.Series(np.exp(pred) - 1)
#     if (i == 0):
#         preds = pred
#     else:
#         preds += pred
# preds /= len(moedels)
# preds[preds < 0] = 0.01

# make lightGBM predictions

# output.to_csv('sub.csv', index=False)

end_datetime = datetime.datetime.now()
print('End at {}'.format(end_datetime.strftime("%Y-%m-%d %H:%M:%S")))
# print('Total testing time: {} seconds'.format((end_datetime-train_end_datetime).total_seconds()))
print('Training + Testing time: {} seconds'.format((end_datetime-start_real).total_seconds()))


# submission.to_csv("./rnn_ridge_submission_best.csv", index=False)
