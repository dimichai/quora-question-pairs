from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def clean_train_set(data):
    # Drop items with NA questions
    data = data.dropna()
    # Get the labels of the kept data
    labels = data['is_duplicate']
    # Drop isDuplicate from train set, not needed
    data = data.drop('is_duplicate', axis=1)
    
    return data, labels

def remove_stopwords(text):
    text = text.split()
    text = [w for w in text if not w in stop_words]
    text = ' '.join(text)

    return text