import pandas as pd
import matplotlib.pyplot as plt
 
# reading the database
data = pd.read_csv("data/tips.csv")
 
# Bar chart 
plt.bar(data['day'], data['tip'])

# Adding Title to the Plot
plt.title("Bar Chart")

 
# Setting the X and Y labels
plt.xlabel('Day')
plt.ylabel('Tip')
 
plt.show()

