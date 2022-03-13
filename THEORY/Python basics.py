# In[Lists & Loops]
    
my_list = [1, 2, 3, "a", "b", "c"]

print("My list is", my_list)
    
print("\nLooping through a list ...")
for item in my_list:
    print("item is", item)
    
print("\nmy_list has", len(my_list), "elements")

print("\nAlternatively, another way to loop through a list...")
for index in range(len(my_list)):
    item = my_list[index]
    print("item is", item)
    
print("\nSlicing a list from beginning...")
for item in my_list[:3]:
    print("item is", item)
    
print("\nSlicing a list to the end...")
for item in my_list[3:]:
    print("item is", item)
    
print("\nSlicing a list in the middle...")
for item in my_list[2:4]:
    print("item is", item)
    
print("\nEnumerating a list...")
for i, item in enumerate(my_list):
    print("item number", i, "is", item)
    
print("\nAnother way to enumerate using a list method...")
for item in my_list:
    index = my_list.index(item)
    print("item", item, "has index", index)
    
# In[For Loops]
    
print("Demonstrating for loop")
for i in [0, 1, 2, 3, 4, 5]:
    print(i)
    
print("\nDemonstrating for loop with range()")
for i in range(6):
    print(i)
    
print("\nThis is an example of a nested loop")
for left_num in range(5):
    for right_number in range(5):
        product = left_num * right_number
        print(left_num, "x", right_number, "=", product)

# In[Control Flow]

Y = 7

if 5 < Y < 10:
    print("Y is between 5 and 10")
else:
    print("Y is not between 5 and 10")
    
X = 4

if X < 5:
    print("X is a small number")
elif X < 20:
    print("X is a medium sized number")
else:
    print("X is a big number")

if True:
    print("True is always True")

if False:
    print("This will never be printed")

# In[Data Types]

print("STRINGS")
my_string_1 = "hello"
my_string_2 = 'world'

my_multiline_string = """
Dear World,
Hello. I am a multiline python string. I'm enclosed in triple quotes. I'd write them here, but that would end the string!
I know! I'll use a slash as an escape character.
Triple quotes look like this: \"\"\"
Sincerely, Python 
"""

newline_character = "\n"
print(my_string_1, my_string_2)
print(my_multiline_string)
print(newline_character)
print("----------")
print(newline_character)

print("NUMBERS")
my_float = 0.5
my_integer = 7
my_negative = -3.5
my_fraction = 1 / 2

does_half_equal_point_five = (my_fraction == my_float)
print("The absolute value of", my_negative, "is", abs(my_negative))
print(my_integer, "squared is equal to", my_integer ** 2)
print("Does", my_fraction, "equal", my_float, "?", does_half_equal_point_five)

# In[List Comprehensions]
numbers_0_to_9 = [x for x in range(10)]
print("Numbers 0 to 9", numbers_0_to_9)

numbers_0_to_9 = []
for x in range(10):
    numbers_0_to_9.append(x)
print("Numbers 0 to 9", numbers_0_to_9)

squares = [x * x for x in range(10)]
print("Squares", squares)

odds = [x for x in range(10) if x % 2 == 1]
print("Odds", odds)

# In[Advanced List Comprehensions]
from collections import namedtuple

Person = namedtuple("Person", ["name", "age", "gender"])
people = [
    Person("Andy", 30, "m"),
    Person("Ping", 1, "m"),
    Person("Tina", 32, "f"),
    Person("Abby", 14, "f"),
    Person("Adah", 13, "f"),
    Person("Sebastian", 42, "m"),
    Person("Carol" , 68, "f")
]

andy = people[0]

print("name:", andy.name)
print("age:", andy.age)
print("gender:", andy.gender)

male_names = [person.name for person in people if person.gender == 'm']
print("\nMale names:", male_names)

teen_names = [p.name for p in people if 13 <= p.age <= 18]
print("Teen names:", teen_names)

# In[Python random library]
import random as rd

a = rd.random()
b = rd.random()
c = rd.random()
print("a is", a)
print("b is", b)
print("c is", c)

# In[Arrays]

import numpy as np

road = np.array(['r', 'r', 'r', 'r', 'r', 's', 'r'])
print("The length of the array is:", len(road))
#print("The length of the array is:" +str(len(road)))

value = road[0]
print("\nValue at index[0] =", value)

value_end = road[-1]
print("\nValue at last index = ", value_end)

equal = (value == value_end)
print("\nAre the first and last values equal?", equal)

length = len(road)
for index in range(0, length):
    value = road[index]
    print("road[" + str(index) + "] = " + str(value))

print("\n")
for index in range(0, length):
    print(str(index))
    if index == 3:
        print("We\'ve reached the middle of the road, leaving the loop")
        break
        
def find_stop_index(road):
    for i in range(len(road) - 1):
        if road[i + 1] != 's':
            stop_index = i - 1
    return stop_index

stop = find_stop_index(road)
print(stop)

# In[Numpy arrays]
import numpy as np

grid = [
        [0, 1, 5],
        [1, 2, 6],
        [2, 3, 7],
        [3, 4, 8]
        ]

print(grid[0])
print(grid[0][0])

np_grid = np.array(grid)

print(np_grid[:, 0])

np_1D = np_grid.reshape(1, -1)
# -1 means automaticallyfit all values into this new array
print(np_1D)

zero_grid = np.zeros((5, 4))
print(zero_grid)

zero_like_grid = np.zeros_like(np_grid)
print(zero_like_grid)

# In[Math in python]

import math

radius = 10.0
diameter = 2 * radius
circumference = 2 * math.pi * radius
area = math.pi * radius ** 2

print("Radius is", radius)
print("Diameter is", diameter)
print("Circumference is", "{0:.2f}".format(circumference))
print("Area is", "{0:.3f}".format(area))

sqr_root_2 = math.sqrt(2)
is_sqr_root_2_an_integer = isinstance(sqr_root_2, int)
print("Is square root two an integer?", is_sqr_root_2_an_integer)
is_sqr_root_2_a_float = isinstance(sqr_root_2, float)
print("Is square root two a float", is_sqr_root_2_a_float)

# In[Plotting scatter]

import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

plt.scatter(x, y)
plt.xlabel("x values")
plt.ylabel("y values")
plt.title("X values vs Y values")
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
# x axis values are indicated with the array above
plt.show()
#Matplotlib is used in a terminal or scripts, plt.show() is a must.
#Matplotlib is used in a IPython shell or a notebook (ex: Kaggle), plt.show() is unnecessary.

# In[Plotting bar chart]

import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

plt.bar(x, y)
plt.xlabel("x values")
plt.ylabel("y values")
plt.title("X values vs Y values")
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
plt.show()

# In[Plotting bar chart #2]
import matplotlib.pyplot as plt

x = ['apples', 'pears', 'bananas', 'grapes', 'melons', 'avocados', 'cherries', 'oranges', 'pumpkins', 'tomatoes']
y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

plt.bar(x, y)
plt.xlabel("x values")
plt.ylabel("y values")
plt.title("X values vs Y values")
plt.xticks(rotation = 70)
plt.show()

# In[Plotting line chart]

import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

plt.plot(x, y)
plt.xlabel("x values")
plt.ylabel("y values")
plt.title("X values vs Y values")
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
plt.show()

# In[Gaussian]

print("Example of squaring elements in a list")
print(np.square([1, 2, 3, 4, 5]))

print("\nExample of taking the square root of a list")
print(np.sqrt([1, 4, 9, 16, 25]))

print("\nExample of taking the cube of a list")
print(np.power([1, 2, 3, 4, 5], 3))

print("\nExample of obtaining the exponential of a list")
print(np.exp([1, 2, 3, 4, 5]))

def gaussian_density(x, mu, sigma):
    return (1/np.sqrt(2 * np.pi * np.square(sigma))) * np.exp(-np.square(x-mu) / (2 * np.square(sigma)))

x = np.linspace(0, 100, 11)
print(x)

def plot_gaussian(x, mu, sigma):
    
    y = gaussian_density(x, mu, sigma)
    
    plt.plot(x, y)
    plt.xlabel('x variable')
    plt.ylabel('probability density function')
    plt.title('Gaussian Probability Density Function')
    plt.show()


x = np.linspace(0, 100, 200)
plot_gaussian(x, 50, 10)

# In[Matrices]

v = [5, 10, 2, 6, 1]
print(v)

mv = [v]

print(mv)

matrix1by2 = [[5, 7]]
matrix2by1 = [[5], [7]]
print(matrix1by2)
print(matrix2by1)

# In[Dictionaries basics]

ticket = {"date" : "2018-12-28",
          "priority" : "high",
          "description" : """Vehicle did not slow down despite
          SLOW
          SCHOOL
          ZONE"""}
          
print(ticket.keys())
print(ticket['description'])
print(ticket.items())

eng_to_spa = {}
print(eng_to_spa)

eng_to_spa["blue"] = "Azul"
eng_to_spa["green"] = "Verde"
print(eng_to_spa)

del eng_to_spa["green"]
print(eng_to_spa)

eng_to_spa["blue"] = "azul"
print(eng_to_spa)

# In[Dictionaries basics example]

eng_to_spa["blue"] = "azul"
eng_to_spa["green"] = "verde"
eng_to_spa["pink"] = "rosa"
eng_to_spa["orange"] = "naranja"
eng_to_spa["brown"] = "marron"
print(eng_to_spa)

for i in eng_to_spa:
    print(i)

print("\n")
for i in eng_to_spa:
    value = eng_to_spa[i]
    print(i, ":", value)

eng_to_spa = {"red" : "rojo",
              "blue" : "azul",
              "green" : "verde",
              "black" : "negro",
              "white" : "blanco",
              "yellow" : "amarillo",
              "orange" : "naranja",
              "pink" : "rosa",
              "purple" : "morado",
              "gray" : "gris"}

def reverse(dictionary):
    new_dict = {}
    for i in dictionary:
        new_dict[dictionary[i]] = i
    return new_dict

print("\n")
print(reverse(eng_to_spa))

fr_to_eng = {"bleu" : "blue",
             "noir" : "black",
             "vert" : "green",
             "violet": "purple",
             "gris" : "gray",
             "rouge" : "red",
             "orange": "orange",
             "rose" : "pink",
             "marron": "brown",
             "jaune" : "yellow",
             "blanc" : "white",
            }

print("\nthere are", len(eng_to_spa), "colors in eng_to_spa")
print("there are", len(fr_to_eng), "colors in fr_to_eng")

eng_to_fr = reverse(fr_to_eng)

print("\n")
for english_word in eng_to_fr:
    french_word = eng_to_fr[english_word]
    print("The french word for", english_word, "is", french_word)

#make spa_to_fr

def spa_to_fr(eng_to_spa, eng_to_fr):
    S2F = {}
        
    for english_word in eng_to_fr:
        if english_word in eng_to_spa:
            french_word = eng_to_fr[english_word]
            spanish_word = eng_to_spa[english_word]
            S2F[spanish_word] = french_word
    return S2F

S2F = spa_to_fr(eng_to_spa, eng_to_fr)
print("\n")
print(S2F)

# In[Dictionary example 2]

def group_by_owners(files): 
    new_dict = {}
    
    for key, value in files.items():
        if value in new_dict:
            new_dict[value].append(key)
        else:
            new_dict[value] = [key]
    
    return new_dict

if __name__ == "__main__":    
    files = {
        'Input.txt': 'Randy',
        'Code.py': 'Stan',
        'Output.txt': 'Randy'
    }   
    print(group_by_owners(files))
# In[Sets]

my_set = set()
print(my_set)

my_set.add("a")
my_set.add("b")
my_set.add("c")
my_set.add(1)
my_set.add(2)
my_set.add(3)

print(my_set)

for element in my_set:
    print(element)

my_set.add(2)
print("\n")
for element in my_set:
    print(element)

my_set.remove("a")
print("\n")
for element in my_set:
    print(element)

# In[Other data structures]

my_tuple = (1, 2, 3)
print(my_tuple)
print(type(my_tuple))

#they are not mutable

t1 = ('a', 'b', 'c')

set_of_tuples = set()
set_of_tuples.add(t1)
set_of_tuples.add(my_tuple)

print(set_of_tuples)

# In[Class]

class Song:
    def __init__(self, name):
        self.name = name
        self.next = None

    def next_song(self, song):
        self.next = song 
    
    def is_repeating_playlist(self):
        """
        :returns: (bool) True if the playlist is repeating, False if not.
        """
        x = set()
        while self is not None:
            x.add(self)
            self = self.next
            if self in x:
                return True
        return False
            
first = Song("Hello")
second = Song("Eye of the tiger")
    
first.next_song(second)
second.next_song(first)
    
print(first.is_repeating_playlist())










