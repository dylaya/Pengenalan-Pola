import pandas as pd

#baca data dari file csv 
data = pd.read_csv('data/Data.csv', sep=';')
print(data)

#baca data dari file txt dengan pemisah tab
print("\nBaca data dari file text")
with open ('data/Data.txt') as data:
    print(data.read())

#baca data dari URL
f = pd.read_csv('http://www.exploredata.net/ftp/Spellman.csv')
print(f)