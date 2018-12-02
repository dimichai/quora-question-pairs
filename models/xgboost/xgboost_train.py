import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import libs.data_cleaner as data_cleaner
import libs.feature_extractor as feature_extractor
# from fuzzywuzzy import fuzz

# Load data
train = pd.read_csv('data/cleaned/train_data.csv')
train_labels = pd.read_csv('data/cleaned/train_labels.csv')

train_labels = train_labels['is_duplicate']

# Create same_words_fraction feature
train_data = pd.DataFrame()
train_data['same_words_fraction'] = train.apply(feature_extractor.same_words_fraction, axis=1, raw=True)

# Jacard similarity
train_data['jacard_similarity'] = train.apply(feature_extractor.get_jaccard_sim, axis=1, raw=True)

# Length features
length_features = feature_extractor.get_length_features(train)
train_data['q1_len'] = length_features['q1_len']
train_data['q2_len'] = length_features['q2_len']
# train_data['len_diff'] = length_features['len_diff']

# train_data['fuzz_qratio'] = train.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
# print('1 out of 7 fuzz features')
# train_data['fuzz_WRatio'] = train.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
# train_data['fuzz_partial_ratio'] = train.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
# print('3 out of 7 fuzz features')
# train_data['fuzz_partial_token_set_ratio'] = train.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
# train_data['fuzz_partial_token_sort_ratio'] = train.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
# print('5 out of 7 fuzz features')
# train_data['fuzz_token_set_ratio'] = train.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
# train_data['fuzz_token_sort_ratio'] = train.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
# print('7 out of 7 fuzz features')

# # Create Vectors
# vectors_q1 = train['question1'].apply(feature_extractor.get_vectors)
# vectors_q2 = train['question2'].apply(feature_extractor.get_vectors)

# cosine_q1 = vectors_q1.apply(feature_extractor.get_cosine_sim)
# cosine_q2 = vectors_q2.apply(feature_extractor.get_cosine_sim)
# # Create Cosine Similarity

# Split train/test data
x_train, x_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.1, random_state=9999)

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test, y_test)

# Set xgboost parameters & train the model
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'error'
params['eta'] = 0.1
params['max_depth'] = 15
params['min_child_weight'] = 2

model = xgb.train(params, dtrain, 400, [(dtrain, 'train'), (dtest, 'validation')], early_stopping_rounds=100, verbose_eval=50)
# model.dump_model('./models/xgboost/xgb_trained.raw.txt')
model.save_model('./models/xgboost/xgb_trained.model')