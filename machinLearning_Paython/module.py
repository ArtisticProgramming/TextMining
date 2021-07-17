import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import naive_bayes
from sklearn.svm import LinearSVC
from sklearn import metrics

class ML_Algorithm:
      def __init__(self, SVM, NB):
            self.SVM = SVM
            self.NB = NB
  
def Learn():
    df = pd.read_csv('./TextFiles/fidiliv4.csv', sep=',')
    df.dropna(inplace=True) 
    blanks = []
    for i,lb,rv in df.itertuples():
        if type(rv)==str:
            if rv.isspace():
                blanks.append(i)
    df.drop(blanks, inplace=True)
    
    X = df['review']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=2000)
    
    
    # Naïve Bayes Model:
    text_clf_nb = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
                         ('clf', naive_bayes()),
    ])
    
    # Linear SVC Model:
    text_clf_lsvc = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
                         ('clf', LinearSVC()),
    ])
    
    # Train both models on the moviereviews.tsv training set:
    text_clf_nb.fit(X_train, y_train)
    text_clf_lsvc.fit(X_train, y_train)

    ml = ML_Algorithm(text_clf_lsvc,text_clf_nb)


    return ml
    
    
