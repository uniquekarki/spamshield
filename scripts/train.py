import numpy as np
import pandas as pd

import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

import pickle


def preprocess(messages):
    lemmatizer = WordNetLemmatizer()
    message = messages.lower()
    stop_words = set(stopwords.words('english'))

    tokens = word_tokenize(message)
    clean_token = [lemmatizer.lemmatize(i) for i in tokens if i not in stop_words]
    messages = ' '.join(clean_token)
    return messages

if __name__ == '__main__':
    df = pd.read_csv('../sms+spam+collection/SMSSpamCollection', sep = '\t', header = None, names = ['label','message'])

    df['label'] = df['label'].apply(lambda x: 1 if x == 'spam' else 0)


    tfidf_vectorizer = TfidfVectorizer(preprocessor=preprocess, tokenizer=str.split)

    X = tfidf_vectorizer.fit_transform(df['message'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    textClassifier = MultinomialNB()
    textClassifier.fit(X_train, y_train)

    y_hat = textClassifier.predict(X_test)
    
    score = accuracy_score(y_hat, y_test)*100
    print(f'Accuracy percentage: {score:.2f}%')

    model_pkl_file = "../models/sms_classifier_model.pkl"  

    with open(model_pkl_file, 'wb') as file:  
        pickle.dump(textClassifier, file)

    vectorizer_pkl_file = "../models/vectorizer.pkl"  

    with open(vectorizer_pkl_file, 'wb') as file:  
        pickle.dump(tfidf_vectorizer, file)