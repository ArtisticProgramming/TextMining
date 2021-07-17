# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 11:41:08 2020

@author: jafar
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from datetime import datetime

npr = pd.read_csv('FILE.csv')
#npr.head()
cv = CountVectorizer(max_df=0.75, min_df=5)
dtm = cv.fit_transform(npr['PreProcessedBody'])

#------LDA------------
LDA = LatentDirichletAllocation(n_components=5,random_state=250)
LDA.fit(dtm)

for index,topic in enumerate(LDA.components_):
    print(f'THE TOP 5 WORDS FOR TOPIC #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-20:]])
    print('\n')
    
    
    
    
    
now = datetime.now()
 # dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)	


topic_results = LDA.transform(dtm)
topic_results.shape
topic_results.argmax(axis=1)
npr['Topic'] = topic_results.argmax(axis=1)
print(npr.head(5))

compression_opts = dict(method='zip', archive_name='out_4_35_neg.csv')  
npr.to_csv('out_4_35_neg.zip', index=False,
          compression=compression_opts)