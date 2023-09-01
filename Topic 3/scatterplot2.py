import pandas as pd
import matplotlib.pyplot as plt

# reading the database
data = pd.read_csv("data/tips.csv")

def showscatterplot(sbx, sby):
    # Scatter plot with day against tip
    plt.scatter(data[sbx], data[sby])
    # Adding Title to the Plot
    plt.title("Scatter Plot")
    # Setting the X and Y labels
    plt.xlabel(sbx)
    plt.ylabel(sby)
    plt.show()

showscatterplot('day','tip')
showscatterplot('smoker','tip')
showscatterplot('size','tip')
showscatterplot('sex','tip')



 
