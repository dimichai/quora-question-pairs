import pandas as pd
import libs.feature_extractor as feature_extractor
import libs.data_cleaner as data_cleaner
import xgboost as xgb
# from fuzzywuzzy import fuzz

# Predict Kaggle's test set
test = pd.read_csv('data/test_data.csv')    

# Clean text - remove not needed information.
test['question1'] = test['question1'].apply(data_cleaner.clean_text)
test['question2'] = test['question2'].apply(data_cleaner.clean_text)

# Clean text - remove stopwords.
test['question1'] = test['question1'].apply(data_cleaner.remove_stopwords)
test['question2'] = test['question2'].apply(data_cleaner.remove_stopwords)

# Feature Creation
test_data = pd.DataFrame()
test_data['same_words_fraction'] = test.apply(feature_extractor.same_words_fraction, axis=1, raw=True)
test_data['jacard_similarity'] = test.apply(feature_extractor.get_jaccard_sim, axis=1, raw=True)
# test_data['fuzz_qratio'] = test.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)

# Length features
length_features = feature_extractor.get_length_features(test)
test_data['q1_len'] = length_features['q1_len']
test_data['q2_len'] = length_features['q2_len']
# test_data['len_diff'] = length_features['len_diff']

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

submission.to_csv('simple_features_xgboost.csv', index=False)