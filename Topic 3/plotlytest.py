import plotly.express as px
import pandas as pd
 
# reading the database
data = pd.read_csv("data/tips.csv")
 
# plotting the scatter chart
fig = px.scatter(data, x="day", y="tip", color='sex')
# showing the plot
fig.show()

# plotting the line chart
fig = px.line(data, y='tip', color='sex')
# showing the plot
fig.show()

# plotting the bar chart
fig = px.bar(data, x='day', y='tip', color='sex')
# showing the plot
fig.show()

# plotting the histogram chart
fig = px.histogram(data, x='total_bill', color='sex')
# showing the plot
fig.show()

