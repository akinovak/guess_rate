from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

from sklearn.naive_bayes import  MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import nltk

import sklearn.metrics as met

from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords

from nltk.tokenize import sent_tokenize, word_tokenize

import numpy as np

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import sklearn.preprocessing as prep

porter = PorterStemmer()
lancaster=LancasterStemmer()

def stemSentence(sentence):
    token_words=sentence.split(" ")
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)


# print(porter.stem(sentence))

df = pd.read_csv('Hotel_Reviews.csv')

positive = df['Positive_Review']

negative = df['Negative_Review']


corpus = []

count = 0

for row in positive :

    processed_row = stemSentence(row)

    corpus.append(processed_row)

    count+= 1

    if count == 3000 :
        break

count = 0

for row in negative :

    processed_row = stemSentence(row)

    corpus.append(processed_row)

    count+= 1

    if count == 3000 :
        break


classes = ['positive' if i < 3000 else 'negative' for i in range(6000)]


vectorizer = TfidfVectorizer(stop_words='english')
x_fitted = vectorizer.fit_transform(corpus)

x_train, x_test, y_train, y_test = train_test_split(x_fitted, classes, stratify = classes, test_size = 0.3)

clf = MultinomialNB()

clf.fit(x_train, y_train)

df_visualisator = pd.DataFrame(clf.feature_count_, index =clf.classes_,  columns = vectorizer.get_feature_names())


hotels = df['Hotel_Name'].unique()

cnt = 0

new_df = []

for h in hotels :

    if cnt == 100 :

        break

    diff_of_com = 0

    tmp_df = df.loc[df['Hotel_Name'] == h]

    average_score = tmp_df['Average_Score'].unique()

    num_reviews = tmp_df['Total_Number_of_Reviews'].unique()
    # print(tmp_df.head())

    for positive in tmp_df['Positive_Review'] :

        
        # positive_text = row['Positive_Review']

        # print(positive)

        x_predict = vectorizer.transform([positive])

        y_predict = clf.predict(x_predict)

        probabilies = clf.predict_proba(x_predict)

        s = pd.Series(probabilies[0], index = clf.classes_)

        diff_of_com += s[1] - s[0]

    for negative in tmp_df['Negative_Review'] :

        x_predict = vectorizer.transform([negative])

        y_predict = clf.predict(x_predict)

        probabilies = clf.predict_proba(x_predict)

        s = pd.Series(probabilies[0], index = clf.classes_)

        diff_of_com += s[1] - s[0]

    
    avg_negative_word_count = np.average(tmp_df['Review_Total_Negative_Word_Counts'])
    avg_positive_word_count = np.average(tmp_df['Review_Total_Positive_Word_Counts'])

    new_df.append([average_score[0], h, num_reviews[0], avg_positive_word_count, avg_negative_word_count, diff_of_com / num_reviews[0]])

    print('For hotel : {}'.format(h))

    cnt += 1
    # print (diff_of_com / num_reviews)

df = pd.DataFrame(new_df, columns = ['Average_Score', 'Hotel_Name', 'Num_Reviews', 'AVG_pos', 'AVG_neg', 'Pos_Negative_Proba'])

lab_enc = prep.LabelEncoder()

# new_df.set_index('Hotel_Name', inplace = True)

# print(new_df.head())

y = df['Average_Score']

# y = lab_enc.fit_transform(y)

y = ['very_bad' if e < 6 else e for e in y]
y = ['bad' if not isinstance(e, str) and e < 7 else e for e in y]
y = ['good' if not isinstance(e, str) and e < 8 else e for e in y]
y = ['very_good' if not isinstance(e, str) and e < 9 else e for e in y]
y = ['excelent' if not isinstance(e, str) and e < 10 else e for e in y]

# y = np.round(y)

features = df.columns[2:].tolist()
x=df[features]

print(x.columns)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# print(x_train)
print(y_train)

# clf = SVC(C=1.0, kernel='linear')
# clf.fit(x_train, y_train)

# Parametri za unakrsnu validacuju
parameters_for_SVC = [{'C': [pow(2,x) for x in range(-6,10,2)],
               'kernel' : ['linear']
               },

              {'C': [pow(2,x) for x in range(-6,10,2)],
               'kernel': ['poly'],
               'degree': [2, 3, 4, 5],
               'gamma': np.arange(0.1, 1.1, 0.1),
               'coef0': np.arange(0, 2, 0.5)
               },

                {'C': [pow(2,x) for x in range(-6,10,2)],
               'kernel' : ['rbf'],
               'gamma': np.arange(0.1, 1.1, 0.1),
               },

               {'C': [pow(2,x) for x in range(-6,10,2)],
               'kernel' : ['sigmoid'],
               'gamma': np.arange(0.1, 1.1, 0.1),
               'coef0': np.arange(0, 2, 0.5)
               }]


parameters_for_KNN = [
                {'n_neighbors': [3, 4, 5],
               'weights' : ['distance', 'uniform'], 
                'p' : [1, 2]
               }]


#Umesto SVC inde KNeighbour Classifier ako se biraju parametri za knn
clf = GridSearchCV(KNeighborsClassifier(), parameters_for_KNN, cv=5, scoring='precision_macro')
clf.fit(x_train, y_train)

# print(clf.best_params_)
print(clf.best_score_)
print(clf.best_params_)

y_predicted = clf.predict(x_train)

print(met.confusion_matrix(y_train, y_predicted))