import pandas as pd
 
# reading the database
data = pd.read_csv("data/tips.csv") #lokasi menyesuaikan
 
# printing the top 10 rows
#display(data.head(10))
print(data.head(10))
