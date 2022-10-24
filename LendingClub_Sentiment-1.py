# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 18:32:54 2022

@author: hp
"""

import pandas as pd
import matplotlib.pyplot as plt
messages = pd.read_csv('D:\EXL/LendingClub1.csv')
data = messages['body']

#Data CLeaning
import re
import nltk
#nltk.download()
#nltk.download('stopwords')



from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
word_n = WordNetLemmatizer()
corpus = []
for i in range(0,len(data)):
    review = re.sub('[^a-zA-Z]',' ',data[i])
    review = review.lower()
    review = review.split()
    review = [word_n.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#TF-IDF vectorizer
tfv = TfidfVectorizer()
#transform
vec_text = tfv.fit_transform(corpus).toarray()
#returns a list of words.
words = tfv.get_feature_names()


#setup kmeans clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2, tol = 0.01, max_iter = 200,random_state=22)    

kmeans.fit(vec_text)
labels = kmeans.labels_


print(pd.DataFrame(labels).value_counts())

per = (7752/10290)*100
per

####trying to find accuracy
data_clustered = pd.concat([pd.DataFrame(corpus),pd.DataFrame({'cluster':labels})],axis = 1)
data_clustered.head()


positive_review = pd.DataFrame(data_clustered[data_clustered['cluster']== 0])
negative_review = pd.DataFrame(data_clustered[data_clustered['cluster']== 1])

positive_review.columns


'''words = []
for i in range(0,len(positive_review)):
    for word in positive_review[0][i]:
        words.append(word)'''


from wordcloud import WordCloud

text_pos = " ".join(word.split()[1] for word in positive_review[0])

text_neg = " ".join(word.split()[1] for word in negative_review[0])

word_cloud_pos = WordCloud(collocations = False, background_color = 'white').generate(text_pos)
plt.imshow(word_cloud_pos, interpolation='bilinear')
plt.axis("off")
plt.show()



word_cloud_neg = WordCloud(collocations = False, background_color = 'white').generate(text_neg)
plt.imshow(word_cloud_neg, interpolation='bilinear')
plt.axis("off")
plt.show()



#this loop transforms the numbers back into words
'''common_words = kmeans.cluster_centers_.argsort()[:,-1:-11:-1]
for num, centroid in enumerate(common_words):
    print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))'''


