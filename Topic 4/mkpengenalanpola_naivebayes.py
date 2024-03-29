# -*- coding: utf-8 -*-
"""MKPengenalanPola_NaiveBayes.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dkCUEtotsuqW8mmgSTKDI-zO7iyX0JMI

Sebelum melakukan klasifikasi, terlebih dahulu kita mengetahui bagaimana karakteristik pola dari data yang kita miliki. Untuk itu, kita akan menyajikan secara deskriptif data tersebut dengan histogram dan boxplot
"""

#Membaca data dan mendapatkan deskrips data
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('data/Iris.csv')
print (data.head(10))
data.describe()

# Menyajikan histogram dari suatu variabel independen
plt.figure(figsize = (10, 7))
x = data["sepal.length"]

plt.hist(x, bins = 20, color = "green")
plt.title("Sepal Length in cm")
plt.xlabel("Sepal_Length_cm")
plt.ylabel("Count")

plt.show()

# Menyajikan boxplot dari dataset

new_data = data[["sepal.length", "sepal.width", "petal.length", "petal.width"]]
print(new_data.head())

plt.figure(figsize = (10, 7))
new_data.boxplot()

plt.show()

"""Setelah mengetahui karakteristik dataset, berikut ini adalah beberapa tahapan yang dapat dilakukan untuk mengimplementsaikan naïve bayes dengan Python"""

#	Melakukan pembacaan data
import pandas as pd
import numpy as np

iris = pd.read_csv("data/Iris.csv")
iris.head()

#  variabel bebas
x = iris.drop(["variety"], axis = 1)
x.head()

#variabel tidak bebas
y = iris["variety"]
y.head()

#	Persiapan melakukan klasifikasi

# separate the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 5)

#import from library
from sklearn.naive_bayes import GaussianNB

# Call Gaussian Naive Bayes
iris_model = GaussianNB()

# Melakukan training
# Insert the training dataset to  Naive Bayes function
NB_train = iris_model.fit(x_train, y_train)

# Melakukan prediksi data testing
# Next step: Prediction the x_test to the model built and save to the  y_pred variable
# show the result of prediction
y_pred = NB_train.predict(x_test)
np.array(y_pred)

# show the y_test based on separation dataset
np.array(y_test)

# Menentukan probabilitas hasil prediksi
# this value will show all probability for each predicted class
NB_train.predict_proba(x_test)

#	Melihat hasil dalam bentuk Matrix Confusi
# show the confusion matrix based on the prediction result
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)

# Menyajikan hasil summary performa klasifikasi
# evaluate performance from the confusion matrix
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))