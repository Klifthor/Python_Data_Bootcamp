'''
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
'''

import numpy as np

'''
arr_0D = np.array(10)
print(arr_0D)

arr_1D = np.array([1, 2, 3, 4, 5])
print(arr_1D)

arr_2D = np.array([[1, 2], [3, 4]])
print(arr_2D)

arr_3D = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(arr_3D)

arr1 = np.array(10)
arr2 = np.array([1, 2, 3, 4, 5])
arr3 = np.array([[1, 2], [3, 4]])
print(arr1.ndim, arr2.ndim, arr3.ndim)

arr4 = np.array([[1, 2, 3], [4, 5, 6]])
print(arr4[1, 1])
print(arr4[0, -1])


arr5 = np.array([10, 20, 30, 40, 50])
print(arr5[1:4])
print(arr5[2:])
print(arr5[:4])


arr6 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
print(arr6[0 : 7 : 2])

arr7 = np.array([[1, 2, 3], [4, 5, 6]])

print(arr7[0, 0:2])

arr8 = np.array([1, 2, 3, 4, 5])
copy = arr8.copy()

copy[0] = 24
print(arr8)
print(copy)

view = arr8.view()
view[0] = 24
print(arr8)
print(view)



arr9 = np.array([[1, 2, 3], [4, 5, 6]])
copy  = arr9.copy()

print(arr9)
print()
print(arr9.shape)

arr10 = np.array([1, 2, 3, 4, 5, 6])
view = arr10.view()

print(arr10)
print()
print(arr10.reshape(2, 3))


arrList = np.array_split(arr10, 3)

for array in arrList:
    print(array)


arr11 = np.array([1, 2, 3, 2, 5, 2])

print(np.where(arr11 == 2))



arr12 = np.array([1, 2, 3, 4, 5, 6])

print(np.where(arr12 % 2 == 0))





arr13 = np.array([2, 5, 1, 3, 6, 4])

print(np.sort(arr13))


arr14 = np.array([True, False, False, True])

print(np.sort(arr14))

arr15 = np.array(["Pasta", "Bean", "Cake", "e", "d", "f", "g", "b", "a", "c"])      #


print(np.sort(arr15))


arr16 = np.array([1, 2, 3, 4 ])
arr17 = np.array([1, 2, 3, 4])
#print(arr17 + arr16)   SAME RESULT
print(np.add(arr16, arr17 ))


arr18 = np.array([10, 20, 30, 40])
arr19 = np.array([1, 2, 3, 4])

print(np.subtract(arr18, arr19))



arr20 = np.array([1, 2, 3, 4])
arr21 = np.array([1, 2, 3, 4])

print(np.multiply(arr20, arr21))



arr22 = np.array([10, 20, 30, 40])
arr23 = np.array([1, 2, 3, 4])
print(np.divide(arr22, arr23))


arr24 = np.array([1, 2, 3, 4])
arr25 = np.array([1, 2, 3, 4])

print(np.power(arr24, arr25))



arr25 = np.array([10, 10, 10, 10])
arr26 = np.array([1, 2, 3, 4])
arr27 = np.array([-1, 2, -3, 4])

print(np.mod(arr25, arr26))
print(np.absolute(arr27))

arr28 = np.array([1.23, 3.45, 6.78])

print(np.trunc(arr28))
print(np.fix(arr28))

print(np.around(arr28, 1))





arr29 = np.array([1.2345, 6.789])

print(np.floor(arr29))
print(np.ceil(arr29))


arr30 = np.array([1, 2, 3, 4])

print(np.log(arr30))
print(np.log2(arr30))
print(np.log10(arr30))



arr30 = np.array([1, 2, 3, 4])
arr31 = np.array([5, 6, 7, 8])

print(np.sum([arr30, arr31]))

# Next, we use axis in order to sum each array: Result: [10 26]
print(np.sum([arr30, arr31], axis=1))

arr32 = np.array([1, 2, 3])

print(np.cumsum(arr32))


arr30 = np.array([1, 2, 3, 4])
arr31 = np.array([5, 6, 7, 8])

print(np.prod([arr30, arr31]))

# We use axis again for the products of each array: Result: [ 24  1680]
print(np.prod([arr30, arr31], axis=1))
print(np.cumprod(arr30))




arr33 = np.array([1, 2, 3, 4, 5])

print(np.lcm.reduce(arr33))

print(np.gcd.reduce(arr33))




arr34 = np.array([np.pi/2, np.pi/3, np.pi/4])

print(np.around(np.sin(arr34), 8))
print(np.around(np.cos(arr34), 8))



arr35 = np.array([90, 180, 360])

arr35 = np.deg2rad(arr35)
print(arr35)

arr35 = np.rad2deg(arr35)

print(arr35)


print(np.hypot(3, 4))

arr28 = np.array([1.23, 3.45, 6.78])

print(np.trunc(arr28))
print(np.fix(arr28))

print(np.around(arr28, 1))



import pandas as pd
x = [23, 48, 19]
my_first_series = pd.Series(x)
print(my_first_series)

print(type(x))

import pandas as pd
data = {
    "Students": ['Emma', 'John', 'Bob'],
    "Grades": [12, 18, 17],
    "Sex":  ['Female', 'Male', 'Male']
}

my_first_dataframe = pd.DataFrame(data)
print(my_first_dataframe['Students'])


my_first_dataframe = pd.DataFrame(data, index=["a", "b", "c"])
first_row = my_first_dataframe.loc["a"]
second_row = my_first_dataframe.iloc[1]
print(first_row)
print(second_row)





import pandas as pd
import numpy as np
data = {
    "Students": ['Emma', 'John', np.nan, 'Bob'],
    "Grades": [12, np.nan, 18, np.nan]
}
my_first_df = pd.DataFrame(data, index=["a", "b", "c", "d"])
#my_first_df["Students"].fillna("No Name", inplace=True)
#my_first_df["Grades"].fillna("No Grade", inplace=True)
df2 = my_first_df.interpolate(method='linear', limit_direction='forward')

print(df2)



import pandas as pd
import numpy as np
data = {
    "Students" : ['Emma', 'John', 'Mary', 'Bob'],
    "Grades" :  [12, np.nan, 18, np.nan]
}
my_first_df = pd.DataFrame(data)
print(my_first_df)
#my_first_df = pd.DataFrame(data, index=["a", "b", "c", "d"])
print(my_first_df)
my_first_df.dropna(inplace=True)
print(my_first_df)




import pandas as pd
s = pd.Series(['workearly', 'e-learning', 'python'])
for index, value in s.items():
    print(f"Index: {index}, Value: {value}")





import pandas as pd
data = {
    "students": ['Emma', 'John'],
    "grades": [12, 19.8]
}
my_first_df = pd.DataFrame(data, index=["a", "b"])
for i,j in my_first_df.iterrows():
    print(i,j)
    print()





import pandas as pd
data = {
    "students": ['Emma', 'John'],
    "grades": [12, 19.8]
}

my_first_df =pd.DataFrame(data, index=["a", "b"])
columns = list(my_first_df)
for i in columns:
    print(my_first_df[i][1])





import pandas as pd
df = pd.read_csv("finance_liquor_sales.csv")
print(df.head(12))


import pandas as pd
df = pd.read_csv("finance_liquor_sales.csv")
print(df.tail())
print(df.info())
print(df.shape)



import pandas as pd
df = pd.read_csv("finance_liquor_sales.csv")
mean = df.mean(numeric_only=True)
print(mean)
median = df.median(numeric_only=True)
print(median)

max_v = df.max(numeric_only=True)
print(max_v)

summary = df.describe()
print(summary)




import pandas as pd
df = pd.read_csv("finance_liquor_sales.csv")
cn = df.groupby('category_name')
print(cn.first())



import pandas as pd
df = pd.read_csv("finance_liquor_sales.csv")
cn2 = df.groupby(['category_name', 'city'])
print(cn2.first())




import pandas as pd
import numpy as np
df = pd.read_csv("finance_liquor_sales.csv")
cn = df.groupby('category_name')
print(cn.aggregate(np.sum))




import pandas as pd
df = pd.read_csv("finance_liquor_sales.csv")
cnc = df.groupby(['category_name', 'city'])

# different aggregation to the columns of a DataFrame, we can pass a dictionary to aggregate
print(cnc.agg({'bottles_sold': 'sum', 'sale_dollars': 'mean'}))



import pandas as pd

df = pd.read_csv("finance_liquor_sales.csv")

ng = df.groupby('vendor_name')
print(ng.filter(lambda x: len(x) >=20))




import pandas as pd
d1 = {'Name': ['Mary', 'John', 'Alice', 'Bob'],
      'Age': [27, 24, 22, 32],
      'Position': ['Data Analyst', 'Trainee', 'QA Tester', 'IT']}
d2 = {'Name': ['Steve', 'Tom', 'Jenny', 'Nick'],
      'Age': [37, 25, 24, 52],
      'Position': ['IT', 'Data Analyst', 'Consultant', 'IT']}
df1 = pd.DataFrame(d1, index=[0, 1, 2, 3])
df2 = pd.DataFrame(d2, index=[4, 5, 6, 7])
result = pd.concat([df1, df2])
print(result)


import pandas as pd
d1 = {'key': ['a', 'b', 'c', 'd'],
      'Name': ['Mary', 'John', 'Alice', 'Bob']}
d2 = {'key': ['a', 'b', 'c', 'd'],
      'Age': [27, 24, 22, 32]}
df1 = pd.DataFrame(d1)
df2 = pd.DataFrame(d2)
result = pd.merge(df1, df2, on='key')
print(result)



import pandas as pd
d1 = {'Name': ['Mary', 'John', 'Alice', 'Bob'],
      'Age': [27, 24, 42, 32]}
d2 = {'Position': ['Data Analyst', 'Trainee', 'QA Tester', 'IT'],
      'Years_of_experience': [5, 1, 10, 3]}
df1 = pd.DataFrame(d1, index=[0, 1, 2, 3])
df2 = pd.DataFrame(d2, index=[0, 2, 3, 4])
result = df1.join(df2, how='inner')
print(result)


# Create and display a one-dimensional array-like object containing this array of data L

import pandas as pd
L = [5, 10, 15, 20, 25]
ds =pd.Series(L)
print(ds)


# Write a Pandas program to convert the first column of a DataFrame as a Series.

import pandas as pd
d = {'col1': [1, 2, 3, 4, 7, 11],
     'col2': [4, 5, 6, 9, 5, 0],
     'col3': [7, 5, 8, 12, 1, 11]}
df = pd.DataFrame(d)
s1 = df.iloc[:, 1]
print("1st column as a Series:")
print(s1)
print(type(s1))



# Write a Pandas program to read data.csv file and print the first 20 rows

import pandas as pd
df = pd.read_csv('data.csv')
print(df.head(20))



# Write a Pandas program to iterate through df DataFrame from previous task

import pandas as pd
df = pd.read_csv('data.csv')
for i,j in df.iterrows():
     print(i,j)




# Practicing around with stuff
import pandas as pd
data = {
    "students": ['Emma', 'John', 'Bob'],
    "grades": [12, 19.8, 18],
    "age": [20, 19, 22],
    "sex": ['Female', 'Male', 'Male']
}

my_first_df = pd.DataFrame(data, index=["a", "b", "c"])
print(my_first_df.sort_index())
print(my_first_df.sort_values(by="grades",ascending=False))

#for i,j in my_first_df.iterrows():
#   print(i,j)
#   print()



import pandas as pd
import numpy as np

data = pd.read_csv('1.supermarket.csv')

#print(data.head())
#print('\nShape of dataset:', data.shape)
#print(data.info())

print(data.columns)

x = data.groupby('item_name')
x = x.sum()
print(x.head(1))




############ KARGA PLOTS LEME



import matplotlib.pyplot as plt
plt.plot([0, 10], [0, 300],'o')

plt.show()

# PLOT!

import matplotlib.pyplot as plt

plt.plot([0, 2, 4, 6, 8, 10], [3, 8, 1, 10, 5, 12])

plt.show()




##### PLOT AGAIN

import matplotlib.pyplot as plt

#plt.plot([1, 2, 4], [3, 8, 1], marker='o')
plt.plot([1, 2, 4], [3, 8, 1], ls='dotted')

plt.show()

##### MORE PLOTS!


import matplotlib.pyplot as plt

plt.plot([0, 10], [0, 300])

#plt.title("Title")
#plt.xlabel("X - Axis")
#plt.ylabel("Y - Axis")

plt.grid()

plt.show()



### PLOTS AGAIN!


import matplotlib.pyplot as plt

plt.subplot(2, 1, 1)
plt.plot([0, 2, 4, 6, 8, 10], [3, 8, 1, 10, 5, 12])

plt.subplot(2, 1, 2)
plt.plot([0, 10], [0, 300])

plt.show()




##### GUESS WHAT!PLOTS!

import matplotlib.pyplot as plt
import numpy as np

x = np.array([99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86])

y = np.array([5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6])

plt.scatter(x, y)

plt.show()



### ANOTHER PLOT EXAMPLE

import matplotlib.pyplot as plt
import numpy as np

x = np.array([99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86])

y = np.array([5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6])

plt.scatter(x, y)

x = np.array([100, 105, 84, 105, 90, 99, 90, 95, 94, 100, 79, 112, 91, 80, 85])

y = np.array([2, 2, 8, 1, 15, 8, 12, 9, 7, 3, 11, 4, 7, 14, 12])

plt.scatter(x, y)
plt.show()



### BARCHART 

import matplotlib.pyplot as plt
import numpy as np

x = np.array(["A", "B", "C", "D"])
y = np.array([6, 5, 1, 10])

plt.bar(x, y)

plt.show()



### PIE PLOT

import matplotlib.pyplot as plt
import numpy as np

mylabels = np.array(["Potatoes", "Bacon", "Tomatoes", "Sausages"])

x = np.array([25, 35, 15, 25])

plt.pie(x, labels=mylabels)
plt.legend()

plt.show()





# Example with plot

import matplotlib.pyplot as plt
import numpy as np

age = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

cardiac_cases = [5, 15, 20, 40, 55, 55, 70, 80, 90, 95]

survival_chances = [99, 99, 90, 90, 80, 75, 60, 50, 30, 25]

plt.xlabel("Age")
plt.ylabel("Percentage")

plt.plot(age, cardiac_cases, color='black', linewidth=2, label= "Cardiac Cases", marker='o',
         markerfacecolor='red', markersize=12)

plt.plot(age, survival_chances, color='yellow', linewidth=3, label='Survival Chances',
         marker='o', markerfacecolor='green', markersize=12)

plt.legend(loc='lower right', ncol=1)

plt.show()


# Example with randomiser & piechart

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

products = np.array([['Apple', "Orange"],
                     ["Beef", "Chicken"],
                     ["Candy", "Chocolate"],
                     ["Fish", "Bread"],
                     ["Eggs", "Bacon"]])

random = np.random.randint(2, size=5)
choices = []

counter = 0
for product in products:
    choices.append(product[random[counter]])
    counter = counter + 1

print(choices)

percentages = []

for i in range(4):
    percentages.append(np.random.randint(25))
percentages.append(100 - np.sum(percentages))

print(percentages)

plt.pie(percentages, labels=choices)
plt.legend(loc='lower right', ncol=1)

plt.show()



###### WEB SCRAPING!


import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('1.supermarket.csv')

q = data.groupby('item_name').quantity.sum()

plt.bar(q.index, q, color=['orange', 'purple', 'yellow', 'red', 'green', 'blue', 'cyan'])
plt.xlabel('Items')
plt.xticks(rotation=6)
plt.ylabel('Number of Times Ordered')
plt.title('Most ordered Supermarket\'s Items')
plt.show()



###### Grabbing the Title of a website
import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/List_of_highest-paid_film_actors"
url_txt = requests.get(url).text

s = BeautifulSoup(url_txt, 'lxml')

#print(s.prettify())

print(s.title)
print(s.title.string)



### Finding all the <a> TAGS


import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/List_of_highest-paid_film_actors"
url_txt = requests.get(url).text
s = BeautifulSoup(url_txt, 'lxml')

tag = s.find_all('a')

print(tag)





#### Finding all the TABLE tags

import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/List_of_highest-paid_film_actors"
url_txt = requests.get(url).text
s = BeautifulSoup(url_txt, 'lxml')

tables = s.find_all('table')

print(tables)




##### Finding all the TABLE LINKS

import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/List_of_highest-paid_film_actors"
url_txt = requests.get(url).text
s = BeautifulSoup(url_txt, 'lxml')

my_table = s.find('table', class_='wikitable sortable plainrowheaders')
table_links = my_table.find_all('a')

print(table_links)






############ Creating a list with Actors and Movies

import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/List_of_highest-paid_film_actors"
url_txt = requests.get(url).text
s = BeautifulSoup(url_txt, 'lxml')

my_table = s.find('table', class_='wikitable sortable plainrowheaders')
table_links = my_table.find_all('a', href=True)

actors = []
for links in table_links:
    actors.append(links.get('title'))
print(actors)

'''


