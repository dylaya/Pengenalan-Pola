import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# reading the database
data = pd.read_csv("data/tips.csv")

# draw lineplot
sns.lineplot(x="sex", y="total_bill", data=data)
# setting the title using Matplotlib
plt.title('Title using Matplotlib Function')
plt.show()


sns.barplot(x='day',y='tip', data=data, hue='sex')
plt.show()

sns.histplot(x='total_bill', data=data, kde=True, hue='sex')
plt.show()

