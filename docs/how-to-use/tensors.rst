.. _doc_getting_started:

Tensors and Broadcasting
=========================

Broadcasting is a powerful feature in NumDot that allows for arithmetic operations on arrays of different shapes. It can be found in all major tensor libraries, such as `NumPy <https://numpy.org>`__, `tensorflow <http://tensorflow.org>`__ and `xtensor <http://xtensor.readthedocs.io>`__,

This guide will help you understand the rules and applications of broadcasting with code examples.

What is Broadcasting?
---------------------
Broadcasting refers to the process of extending the shapes of arrays during arithmetic operations without actually copying data. It enables NumDot to perform element-wise operations efficiently on arrays of different shapes.

Basic Broadcasting Rules
------------------------
The general rules of broadcasting are as follows:

1. The dimensions of the arrays are compared element-wise, starting from the trailing dimensions.
2. If the dimensions are equal, they are compatible.
3. If one of the dimensions is 1, it will be broadcast to match the other dimension.

Examples
--------
Here are some examples to make the concept clearer:

.. code-block:: gdscript

    # Example 1: Adding a scalar to an array
    var arr1 := nd.array([1, 2, 3])
    var scalar := 2
    var result1 := nd.add(arr1, scalar)
    print(result1)          # Outputs: { 3, 4, 5 }

    # Example 2: Adding a 1D array to a 2D array
    var arr2 := nd.array([[1, 2, 3], [4, 5, 6]])
    var arr3 := nd.array([1, 2, 3])
    var result2 := nd.add(arr2, arr3)
    print(result2)          # Outputs:
                            # {{ 2, 4, 6 },
                            #  { 5, 7, 9 }}

    # Example 3: Adding arrays with incompatible shapes
    var arr4 := nd.array([1, 2, 3])
    var arr5 := nd.array([1, 2])

    var result3 := nd.add(arr4, arr5) # Results in a broadcast error

Practical Applications
----------------------
Broadcasting is particularly useful in various scenarios, such as:

.. code-block:: gdscript

    # Normalizing data
    var data := nd.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    var mean := nd.mean(data, 0)
    var std := nd.std(data, 0)
    var normalized_data := nd.divide(nd.subtract(data, mean), std)
    print(normalized_data)   # Outputs:
                             # {{-1.22474487, -1.22474487, -1.22474487}}
                             #  { 0.        ,  0.        ,  0.        }
                             #  { 1.22474487,  1.22474487 , 1.22474487}}

    # Applying a filter (row-wise operation)
    var arr6 := nd.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    var mask := nd.array([1, 0, 1])
    var filtered := nd.multiply(arr6, mask.get(null, &"newaxis"))
    print(filtered)          # Outputs:
                             # {{1, 2, 3}
                             #  {0, 0, 0}
                             #  {7, 8, 9}}
