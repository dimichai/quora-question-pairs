"""
The original competition dataset contains empty questions and questions containing only special characters.
In this module we get rid of them and create new .csv files with the train set and the labels.
"""

import pandas as pd
from data_cleaner import clean_text, remove_stopwords

# Load data
train = pd.read_csv('data/train_data.csv')
train_labels = pd.read_csv('data/train_labels.csv')

# Merge label with main data frame, because we will remove training data after cleaning.
train['is_duplicate'] = train_labels['is_duplicate']
# Drop items with NA questions
train = train.dropna()

# Clean text - remove not needed information.
train['question1'] = train['question1'].apply(clean_text)
train['question2'] = train['question2'].apply(clean_text)

# Clean text - remove stopwords.
train['question1'] = train['question1'].apply(remove_stopwords)
train['question2'] = train['question2'].apply(remove_stopwords)

# Some text only contained not-needed information, get rid of it.
train = train[(train.question1 != ' ') & (train.question2 != ' ')]
train = train[(train.question1 != '') & (train.question2 != '')]

# Get the labels of the kept data
train_labels = train[['id', 'is_duplicate']]

# Drop isDuplicate from train set, not needed
train = train.drop('is_duplicate', axis=1)

# Save under ./data/cleaned
train.to_csv('./data/cleaned/train_data.csv', index=False)
train_labels.to_csv('./data/cleaned/train_labels.csv', index=False)