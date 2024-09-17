.. _doc_getting_started:

Getting Started
=========================

First, ensure you have :ref:`NumDot installed in your project <doc_how_to_install>`.

Then, jump into the Godot editor, and create a new node with a script attached in which you can try operations.

Creating Arrays
---------------
NumDot provides the :ref:`NDArray <class_NDArray>` type to represent vectors, matrices, and tensors.

.. code-block:: gdscript

    # Creating a 1D array
    var arr1 := nd.array([1, 2, 3, 4, 5])
    print(arr1)

    # Creating a 2D array
    var arr2 := nd.array([[1, 2], [3, 4], [5, 6]])
    print(arr2)

Array Properties
----------------
NumDot arrays have several properties that give useful information about the array.

.. code-block:: gdscript

    print(arr1.shape())      # Outputs: [5]
    print(arr2.shape())      # Outputs: [3, 2]
    print(arr1.ndim())       # Outputs: 1
    print(arr2.ndim())       # Outputs: 2
    print(arr1.dtype())      # Outputs: Int64

Indexing and Slicing
--------------------
NumDot arrays can be indexed and sliced similarly to godot Arrays, but with more powerful capabilities.

.. code-block:: gdscript

    # Indexing
    print(arr1.get(0))         # Outputs: 1
    print(arr2.get(1, 1))      # Outputs: 4

    # Slicing
    print(arr1.get(nd.range(1, 4)))   # Outputs: { 2, 3, 4 }
    print(arr2.get(null, 1))          # Outputs: { 2, 4, 6 }

Common Operations
-----------------
NumDot supports a wealth of operations on arrays. You will find most functions in the global class :ref:`nd <class_nd>`:. Here are a few examples:

.. code-block:: gdscript

    # Basic arithmetic
    var arr3 := nd.add(arr1, 5)
    print(arr3)            # Outputs: { 6, 7, 8, 9, 10 }

    # Element-wise operations
    var arr4 := nd.multiply(arr1, arr1)
    print(arr4)            # Outputs: { 1, 4, 9, 16, 25 }

Useful Functions
----------------
NumDot includes several utility functions that are crucial for data manipulation and scientific computation:

.. code-block:: gdscript

    # Linear space vector
    var arr6 := nd.linspace(0, 10, 5)
    print(arr6)            # Outputs: { 0., 2.5, 5., 7.5, 10. }

    # Aggregation functions
    print(nd.sum(arr1))    # Outputs: 15
    print(nd.mean(arr1))   # Outputs: 3.0
    print(nd.std(arr1))    # Outputs: 1.4142135623730951

