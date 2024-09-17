.. _doc_numpy_xtensor_numdot:

NumPy, xtensor, and NumDot
==========================

NumDot attempts to recreate `NumPy <https://numpy.org>`_-like behavior, and uses `xtensor <https://github.com/xtensor-stack/xtensor>`_ under the hood.
While most things are consistent across all three libraries, differences exist.

Fundamental Comparison
----------------------

- NumDot functions always return tensors, to avoid accidentally promoting dtypes. NumPy returned types like ``np.float32`` can be used like primitive numbers.
- NumDot does not overload operators like ``+`` and ``*``, due to a `gdscript limitation <https://github.com/godotengine/godot-proposals/issues/8383>`_.
- NumDot does not support subscripts (``a[start:stop:step]``, due to a `gdscript limitation <https://github.com/Ivorforce/NumDot/issues/6>`_. Instead, use ``a.get(nd.range(start, stop, step)``.
- NumPy supports keyword arguments. NumDot only supports ordered arguments, due to a `gdscript limitation <https://github.com/Ivorforce/NumDot/issues/10>`_.
- ``xtensor`` operations are lazy views. NumPy and NumDot operations evaluate immediately, with the exception of strideable views.

Function Cheat Sheet
--------------------

.. csv-table:: NumPy, xtensor and NumDot Cheat Sheet
   :file: numpy-xtensor-numdot.csv
   :widths: 25, 25, 25, 25
   :delim: ;
   :header-rows: 1

Haven't found what you need? Try :ref:`nd <class_nd>`!
