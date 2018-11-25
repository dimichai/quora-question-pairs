import pandas as pd
import libs.feature_extractor as feature_extractor
import libs.data_cleaner as data_cleaner
import xgboost as xgb

# Predict Kaggle's test set
test = pd.read_csv('data/test_data.csv')

# Clean text - remove not needed information.
test['question1'] = test['question1'].apply(data_cleaner.clean_text)
test['question2'] = test['question2'].apply(data_cleaner.clean_text)

# Clean text - remove stopwords.
test['question1'] = test['question1'].apply(data_cleaner.remove_stopwords)
test['question2'] = test['question2'].apply(data_cleaner.remove_stopwords)

# Some text only contained not-needed information, get rid of it.
test = test[(test.question1 != ' ') & (test.question2 != ' ')]
test = test[(test.question1 != '') & (test.question2 != '')]

test_data = pd.DataFrame()
test_data['same_words_fraction'] = test.apply(feature_extractor.same_words_fraction, axis=1, raw=True)
test_data['jacard_similarity'] = test.apply(feature_extractor.get_jaccard_sim, axis=1, raw=True)

# Load the model
model = xgb.Booster()
model.load_model('./models/xgboost/xgb_trained.model')

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