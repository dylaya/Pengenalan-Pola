import numpy as np
import pandas as pd
# Vektor
# vektor python numpy with range value

print("vektor default python\n")
a = np.arange(1,20,1)
b = np.arange(1,20,2)

print (" \n vektor via numpy \n")

# vektor via numpy 
c = np.array ([1,2,3,4,5])
d = np.array ([1.5, 2.5, 5, 6, 7])

print(a)
print(b)
print(a.ndim)
print(a.shape)

# Matrix
# mengubah dari 1D menjadi matrik 2D 
a = np.arange(1,21,1)
c = a.reshape((4,5))
print(c)

# List
list1 = ["apple", "banana", "cherry"]
list2 = [1, 5, 7, 9, 3]
list3 = [True, False, False]
list4 = ["abc", 34, True, 40, "male"]

print(list1)

# Data frame
df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                        columns=['a', 'b', 'c'])

print(df) 

