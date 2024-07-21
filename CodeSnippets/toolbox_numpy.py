


ndarray.ndim
ndarray.shape
ndarray.size

np.ones(( 2, 3), dtype=int)
np.floor(10 * rg.random((2, 2)))

np.arange(15).reshape(5,3)
np.arange(10, 30 , 5)
 # 9 numbers from 0 to 2
 np.Iinspace(0, 2, 9)

np. set_printoptions (threshold=sys . maxsize)

a < 35
array([True, True, False, False])

A * B # element wise product
A @ B # dot product
A.dot(B) # dot product


a.sum(axis=0) # sum each column
a.min(axis=1) # min each row

# from start to position 6, exclusive, set every 2nd element to 1000 
a[:6:2]  = 1000
a[0:6:2] = 1000

use indeces of each element i,j as input to f
def f(x,y):
	    return 10 * x + y
np.fromfunction(f, (5,4) , dtype =int)


x[1, ...] the same as x[1, 2, :, :, :]

# flatten array
a.flat
a.ravel()
for item in a.flat:

# stacking arrays
a = np.array()
b = np.array()
np.vstack((a, b))
np.hstack((a, b))

# splitting arrays into 3 pars
np.hsplit(a,3) 

# check equal arrays 
b is a

# copy array elements 
b = a.copy()

# to delete array ofter copyin
b = a[:100].copy()
del a

# indexing with array
a[[1, 3, 4]] = 0
a[[1, 3, 4]] += 1
a[[1, 3, 4]] = [1, 2, 3]

# indexing array with boolean array
a = np.arange(12).reshape( 3, 4)
b = a > 4
b # ' b' is a boolean ivit/i ' a' ' s shape
array( [ [ False, False, False, False ],
[ False, True, True, True],
[ True, True, True, True ] ] )
a[ b] # Id array with the selected elements
array( [ 5, 6, 7, 8, 9, 10, 11] )  





# tensor is a multidimenstional arry designed for keras for better performance
# scalar tensor contains only one number

# vector data --> 2D (samples , features)
# timestamp is usually the second item
# timeseries data --> 3D (samples, timestamp , features) 

# images --> 4D (samples , height, width , color_channels)
# grayscale image --> ( samples , height , width , 1)
# colored image --> (samples , height , width , 3)
# videos --> 5D (samples , frames , height, width , channels)

import numpy as np

def show(item):
    # variable name
    print("\n".join(["#" * 15 + "\nITEM: " + name for name, value in globals().items() if value is item]))

    print("type: " + str(item.dtype))
    print("dimentions: " + str(item.ndim))
    print("shape: " + str(item.shape))
    print("size: " + str(item.size))

x = np.array(12)
y = np.array([[12 , 3 , 6 , 14] , [2 , 4 , 3 , 5] ] )
show(x)
show(y)

# code to convert np array to anther type
x = x.astype(float)


def basic_tools():
    # slicing
    x[index] # Access elements of the array by index.
    x[start:end:step] # Slice the array to extract a subset of elements.
    x[:, col_index] # Access a specific column of a 2D array.
    x[row_index, :] # Access a specific row of a 2D array.

    x[10:100]
    # is the same as
    x[10 : 100 , : , :]

    # select 14 X 14 pixels in the bottom-right corner of all images
    x[:, -14: , -14:]

    # Define a boolean mask
    mask = arr % 2 == 0  # Select even numbers
    x[mask] # Use a boolean mask to filter elements of the array.

    # other methods 
    np.array() # Create an array from a Python list or tuple.
    np.arange() # Create an array with evenly spaced values within a given interval.
    np.zeros() # Create an array filled with zeros.
    np.ones() # Create an array filled with ones.
    np.random.rand() # Create an array with random values from a uniform distribution.
    np.random.randint() # Generate random integers.
    np.random.randn() # Generate random numbers from a standard normal distribution.
    np.random.choice() # Generate a random sample from a given 1D array.

    np.sum() # Sum of array elements.
    np.mean() # Mean of array elements.
    np.std() # Standard deviation of array elements.
    np.min() # Minimum value in the array.
    np.max() # Maximum value in the array.
    np.dot() # Dot product of two arrays.

    x.reshape() # Reshape the array into a new shape.
    x.flatten() #  Flatten the array into a 1D array.
    np.concatenate() # Concatenate arrays along a specified axis.


A = np.array([1.00, 2.00, 3.00])
B = np.array([1.00, 2.00, 3.01])
np.array_equal(A, B)
# False
np.allclose(A, B, atol=0.01)
# True
