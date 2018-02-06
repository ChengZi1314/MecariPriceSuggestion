from datetime import datetime
start_real = datetime.now()
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import gc
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten, Activation
# from keras.layers import Bidirectional
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
import math
# set seed
np.random.seed(123)

def rmsle(Y, Y_pred):
    assert Y.shape == Y_pred.shape
    return np.sqrt(np.mean(np.square(Y_pred - Y )))

train_df = pd.read_table(r'D:\MercariPriceSuggestion\train.tsv')
test_df = pd.read_table(r'D:\MercariPriceSuggestion\test.tsv')

# remove low prices
train_df = train_df.drop(train_df[(train_df.price < 3.0)].index)
# train_df.shape            #(1481658, 8)

train_df = train_df.drop(train_df[(train_df.price > 2000)].index)
# train_df.shape            #(693359, 7)



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
train_df, dev_df = train_test_split(train_df, random_state=123, train_size=0.99)

# Calculate number of train/dev/test examples.
n_trains = train_df.shape[0]
n_devs = dev_df.shape[0]
n_tests = test_df.shape[0]
print("Training on", n_trains, "examples")
print("Validating on", n_devs, "examples")
print("Testing on", n_tests, "examples")


# # ---------------------------------------------RNN---------------------------------------------------------------------#

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


#Processing categorical data
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

# Process text data

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

# full_df['seq_name'][:5]

# Define constants to use when define RNN model
MAX_NAME_SEQ = 10 #17
MAX_ITEM_DESC_SEQ = 75 # 269
MAX_CATEGORY_SEQ = 8 #8
MAX_TEXT = np.max([
    np.max(full_df.seq_name.max()),
    np.max(full_df.seq_item_description.max()),
# np.max(full_df.seq_category.max()),
]) + 100
MAX_CATEGORY = np.max(full_df.category.max()) + 1
MAX_BRAND = np.max(full_df.brand_name.max()) + 1
MAX_CONDITION = np.max(full_df.item_condition_id.max()) + 1
MAX_DESC_LEN = np.max(full_df.desc_len.max()) + 1
MAX_NAME_LEN = np.max(full_df.name_len.max()) + 1
MAX_SUBCAT_0 = np.max(full_df.subcat_0.max()) + 1
MAX_SUBCAT_1 = np.max(full_df.subcat_1.max()) + 1
MAX_SUBCAT_2 = np.max(full_df.subcat_2.max()) + 1

# Get data for RNN model
def get_rnn_data(dataset):
    X = {
        'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ),
        'item_desc': pad_sequences(dataset.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ),
        'brand_name': np.array(dataset.brand_name),
        'category': np.array(dataset.category),
        # 'category_name': pad_sequences(dataset.seq_category, maxlen=MAX_CATEGORY_SEQ),
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

# Fit RNN model to train data

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
        validation_data=(X_dev, Y_dev), verbose=1,
)

# Evaluate RNN model on dev data

print("Evaluating the model on validation data...")
Y_dev_preds_rnn = rnn_model.predict(X_dev, batch_size=BATCH_SIZE)
print(" RMSLE error:", rmsle(Y_dev, Y_dev_preds_rnn))

# Make prediction for test data

rnn_preds = rnn_model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
rnn_preds = np.expm1(rnn_preds)

out = pd.DataFrame(rnn_preds)
out.to_csv('./out.csv')
print('to csv completed!')
del rnn_model, X_train,Y_train
# -------------------------------------------LightGBM------------------------------------------------------------------#


median_price = train_df['price'].median()
mean_price = train_df['price'].mean()
exclude_cols = ['name', 'category_name', 'brand_name', 'item_description', 'subcat_0', 'subcat_1', 'subcat_2']


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


train_df['subcat_0'] = train_df['category_name'].apply(category_detail_1)
train_df['subcat_1'] = train_df['category_name'].apply(category_detail_2)
train_df['subcat_2'] = train_df['category_name'].apply(category_detail_3)

test_df['subcat_0'] = test_df['category_name'].apply(category_detail_1)
test_df['subcat_1'] = test_df['category_name'].apply(category_detail_2)
test_df['subcat_2'] = test_df['category_name'].apply(category_detail_3)

for col in ['category_name', 'brand_name', 'subcat_0', 'subcat_1', 'subcat_2']:
    df_train, df_test = category_features(train_df, test_df, col)

for col in ['category_name', 'brand_name', 'subcat_0', 'subcat_1', 'subcat_2', 'item_condition_id']:
    for type in ['median', 'mean', 'max', 'min']:
        train_df, test_df = price_features(train_df, test_df, col, type)

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
print("////////////////////////////////////////////")
print(df_train)
for i in range(4):
    train_X, valid_X, train_y, valid_y = train_test_split(df_train, target_train, test_size=0.1, random_state=i)
    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(valid_X, label=valid_y)
    watchlist = [d_train, d_valid]

    model = lgb.train(params, train_set=d_train, num_boost_round=240, valid_sets=watchlist, \
                      early_stopping_rounds=20, verbose_eval=10, categorical_feature=cat_features)
    moedels.append(model)
    Y_dev_preds_lgb = model.predict(dev_df.drop(['train_id', 'price'] + exclude_cols, axis=1))
    # ax = lgb.plot_importance(model)
    # plt.tight_layout()
    # plt.savefig('feature_importance_{}.png'.format(i))

for i, model in enumerate(moedels):
    pred = model.predict(df_test.drop(['test_id'] + exclude_cols, axis=1))
    # print("///////////////////////")
    # print(pred)
    # pred = pred.reshape(-1,1)
    # print("///////////////////////////////")
    # print(pred)
    pred = pd.Series(np.exp(pred) - 1)
    if (i == 0):
        LGB_pred = pred
    else:
        LGB_pred += LGB_pred
LGB_pred /= len(moedels)
LGB_pred[LGB_pred < 0] = 0.01

del model,train_X,train_y
# ------------------------------------------------------Ridge Model ---------------------------------------------------#
# Ridge Models
full_df = pd.concat([train_df, dev_df, test_df])
print("//???????????????????????????????????????????")
print(full_df[:10])
print("Handling missing values...")
full_df['category_name'] = full_df['category_name'].fillna('missing').astype(str)
full_df['subcat_0'] = full_df['subcat_0'].astype(str)
full_df['subcat_1'] = full_df['subcat_1'].astype(str)
full_df['subcat_2'] = full_df['subcat_2'].astype(str)
full_df['brand_name'] = full_df['brand_name'].fillna('missing').astype(str)
full_df['shipping'] = full_df['shipping'].astype(str)
full_df['item_condition_id'] = full_df['item_condition_id'].astype(str)
full_df['desc_len'] = full_df['desc_len'].astype(str)
full_df['name_len'] = full_df['name_len'].astype(str)
full_df['item_description'] = full_df['item_description'].fillna('No description yet').astype(str)
print("///////////////////////////////////////////////////////")
print(full_df[:10])
print("Vectorizing data...")
default_preprocessor = CountVectorizer().build_preprocessor()


def build_preprocessor(field):
    field_idx = list(full_df.columns).index(field)
    return lambda x: default_preprocessor(x[field_idx])

vectorizer = FeatureUnion([
    ('name', CountVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        preprocessor=build_preprocessor('name'))),
#     ('category_name', CountVectorizer(
#         token_pattern='.+',
#         preprocessor=build_preprocessor('category_name'))),
    ('subcat_0', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('subcat_0'))),
    ('subcat_1', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('subcat_1'))),
    ('subcat_2', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('subcat_2'))),
    ('brand_name', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('brand_name'))),
    ('shipping', CountVectorizer(
        token_pattern='\d+',
        preprocessor=build_preprocessor('shipping'))),
    ('item_condition_id', CountVectorizer(
        token_pattern='\d+',
        preprocessor=build_preprocessor('item_condition_id'))),
    ('desc_len', CountVectorizer(
        token_pattern='\d+',
        preprocessor=build_preprocessor('desc_len'))),
    ('name_len', CountVectorizer(
        token_pattern='\d+',
        preprocessor=build_preprocessor('name_len'))),
    ('item_description', TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=100000,
        preprocessor=build_preprocessor('item_description'))),
])
print("///////////////////////////////////////////////////////////////////////////////////////////////////")
print(full_df[:10])
X = vectorizer.fit_transform(full_df.values)
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print(X[:10])
X_train = X[:n_trains]
Y_train = train_df.target.values.reshape(-1, 1)

X_dev = X[n_trains:n_trains+n_devs]
Y_dev = dev_df.target.values.reshape(-1, 1)

X_test = X[n_trains+n_devs:]
print(X.shape, X_train.shape, X_dev.shape, X_test.shape)

print("Fitting Ridge model on training examples...")
ridge_model = Ridge(
    solver='auto', fit_intercept=True, alpha=1.0,
    max_iter=100, normalize=False, tol=0.05, random_state = 1,
)
# ridge_modelCV = RidgeCV(
#     fit_intercept=True, alphas=[5.0],
#     normalize=False, cv = 2, scoring='neg_mean_squared_error',
# )

ridge_model.fit(X_train, Y_train)
# ridge_modelCV.fit(X_train, Y_train)

Y_dev_preds_ridge = ridge_model.predict(X_dev)
Y_dev_preds_ridge = Y_dev_preds_ridge.reshape(-1, 1)
print("RMSL error on dev set:", rmsle(Y_dev, Y_dev_preds_ridge))

# Y_dev_preds_ridgeCV = ridge_modelCV.predict(X_dev)
# Y_dev_preds_ridgeCV = Y_dev_preds_ridgeCV.reshape(-1, 1)
# print("CV RMSL error on dev set:", rmsle(Y_dev, Y_dev_preds_ridgeCV))

ridge_preds = ridge_model.predict(X_test)
ridge_preds = np.expm1(ridge_preds)
# ridgeCV_preds = ridge_modelCV.predict(X_test)
# ridgeCV_preds = np.expm1(ridgeCV_preds)

def aggregate_predicts3(Y1, Y2, Y3, ratio1, ratio2):
    assert Y1.shape == Y2.shape
    return Y1 * ratio1 + Y2 * ratio2 + Y3 * (1.0 - ratio1-ratio2)

# Y_dev_preds = aggregate_predicts3(Y_dev_preds_rnn, Y_dev_preds_ridgeCV, Y_dev_preds_ridge, 0.4, 0.3)
# print("RMSL error for RNN + Ridge + RidgeCV on dev set:", rmsle(Y_dev, Y_dev_preds))


#ratio optimum finder for 3 models
best1 = 0
best2 = 0
lowest = 0.99
for i in range(100):
    for j in range(100):
        r = i*0.01
        r2 = j*0.01
        if r+r2 < 1.0:
            Y_dev_preds = aggregate_predicts3(Y_dev_preds_rnn, Y_dev_preds_lgb, Y_dev_preds_ridge, r, r2)
            fpred = rmsle(Y_dev, Y_dev_preds)
            if fpred < lowest:
                best1 = r
                best2 = r2
                lowest = fpred
            print(str(r)+"-RMSL error for RNN + Ridge + RidgeCV on dev set:", fpred)
Y_dev_preds = aggregate_predicts3(Y_dev_preds_rnn, Y_dev_preds_lgb, Y_dev_preds_ridge, best1, best2)
# Y_dev_preds = Y_dev_preds_rnn*0.56 + Y_dev_preds_ridge*0.44
# print(best1)
# print(best2)
print("(Best) RMSL error for RNN + Ridge + RidgeCV on dev set:", rmsle(Y_dev, Y_dev_preds))

# best predicted submission
preds = aggregate_predicts3(rnn_preds, LGB_pred, ridge_preds, best1, best2)
submission = pd.DataFrame({
        "test_id": test_df.test_id,
        "price": preds.reshape(-1),
})
submission.to_csv("./rnn_ridge_submission_best.csv", index=False)
