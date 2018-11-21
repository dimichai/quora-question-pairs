import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import dataclean
from feature_extractor import same_words_fraction

# Load data
train = pd.read_csv('data/train_data.csv')
train_labels = pd.read_csv('data/train_labels.csv')
test = pd.read_csv('data/test_data.csv')

train['is_duplicate'] = train_labels['is_duplicate']

# Clean train data set
train, train_labels = dataclean.clean_train_set(train)
# Remove stopwords
train['question1'] = train['question1'].apply(dataclean.remove_stopwords)
train['question2'] = train['question2'].apply(dataclean.remove_stopwords)

# Create same_words_fraction feature
train_data = pd.DataFrame()
train_data['same_words_fraction'] = train.apply(same_words_fraction, axis=1, raw=True)

# Split train/test data
x_train, x_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.1, random_state=9999)

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test, y_test)

# Set xgboost parameters & train the model
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'error'
params['eta'] = 0.3
params['max_depth'] = 4
params['min_child_weight'] = 1

model = xgb.train(params, dtrain, 100, [(dtest, 'validation')], early_stopping_rounds=50, verbose_eval=10)

# Predict Kaggle's test set
test_data = pd.DataFrame()
test_data['same_words_fraction'] = test.apply(same_words_fraction, axis=1, raw=True)
dsubmission = xgb.DMatrix(test_data)
predictions = model.predict(dsubmission)

submission = pd.DataFrame()
submission['test_id'] = test['test_id']
submission['is_duplicate'] = predictions

# Convert logistic probabilities to binary classifications
submission.loc[submission.is_duplicate < 0.5, 'is_duplicate'] = 0                                                                                                                                                             
submission.loc[submission.is_duplicate >= 0.5, 'is_duplicate'] = 1
# Convert to integer
submission['is_duplicate'] = submission['is_duplicate'].astype(int)

submission.to_csv('single_feature_xgboost.csv', index=False)
