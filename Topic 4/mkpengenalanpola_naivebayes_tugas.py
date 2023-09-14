# Membaca data dan mendapatkan deskripsi data
import matplotlib.pyplot as plt
import pandas as pd
import zipfile

DATADIR = 'data/'
df_zip = zipfile.ZipFile(DATADIR + 'train.csv.zip')
df = pd.read_csv(df_zip.open('train.csv'))

print (df.head(10))
df.describe()

# melihat jumlah data kosong di setiap kolom
df.isnull().sum()

# isi author kosong dengan anonymous
df['author'].fillna('Anonymous', inplace = True)

# hapus data yang kosong
df.dropna(inplace = True)

# ambil variabel independen
X = df.drop('label', axis = 1)
X.head()

# ambil variabel dependen
Y = df['label']
Y.head()

# cek masih ada data kosong
df.isnull().sum()

"""# Text Processing"""

# duplikasi data dan atur ulang index
messages = X.copy()
messages.reset_index(inplace = True)

import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
corpus = []

#hapus stopword dari kolom tertentu
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['text'][i])
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# tampilkan hasil
corpus

# Terapkan TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features = 5000, ngram_range = (1,3))
X = tfidf.fit_transform(corpus).toarray()

# melihat fitur yang dibuat
tfidf.get_feature_names_out()[:10]

# melihat parameter tfidf
tfidf.get_params()

# membagi data train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
print(x_train.shape,' ',y_train.shape)
print(x_test.shape,' ',y_test.shape)

# tampilkan data x_train di dataframe
df = pd.DataFrame(x_train, columns = tfidf.get_feature_names_out())
df.head()

# fungsi plot gambar confusion matrix
import numpy as np
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

"""# Multinominal Naive Bayes"""

from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB

# membuat obyek Multinomial Naive Bayes
classifier = MultinomialNB()

# fit data ke model
classifier.fit(x_train, y_train)
pred = classifier.predict(x_test)

# tampilkan akurasi
score = metrics.accuracy_score(y_test, pred)
print("Accuracy: %0.3f" % score)

# tampilkan confusion matrix
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes = ['FAKE','REAL'])

# tampilkan hasil klasifikasi
from sklearn.metrics import classification_report
print(classification_report(y_test, pred))

"""# Bernoulli Naive Bayes"""

from sklearn.naive_bayes import BernoulliNB

# buat obyek bernoulli naive bayes
classifier = BernoulliNB(alpha=0.1)

# fit  data ke model
classifier.fit(x_train, y_train)
pred = classifier.predict(x_test)

# tampilkan akurasi
score = metrics.accuracy_score(y_test, pred)
print("Accuracy: %0.3f" % score)

#tampilkan confusion matrix
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes = ['FAKE','REAL'])

# tampilkan hasil klasifikasi
print(classification_report(y_test, pred))

"""# Passive Aggressive"""

from sklearn.linear_model import PassiveAggressiveClassifier

# buat object PassiveAggressiveClassifier
classifier = PassiveAggressiveClassifier(max_iter = 50)

# fit data ke model
classifier.fit(x_train, y_train)
pred = classifier.predict(x_test)

# tampilkan akurasi
score = metrics.accuracy_score(y_test, pred)
print("Accuracy: %0.3f" % score)

#tampilkan confusion matrix
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes = ['FAKE','REAL'])

# tampilkan hasil klasifikasi
print(classification_report(y_test, pred))