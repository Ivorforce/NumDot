.. _doc_changelog:

Changelog
=========

Here you will find the release notes for each version of the library. Each section includes information about changes, improvements, and bug fixes.

.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

Upcoming Changes (main branch)
------------------------------
**Added**

- Bounds checks are now enabled everywhere.
- Negative indices are now supported everywhere.
- Added boolean mask indexing, e.g. ``a.set(5, nd.greater(a, 5))``.
- Added index list indexing, e.g. ``a.set(5, [[0, 1], [4, 2]])``.
- Added ``array.copy()`` and ``nd.copy(array)`` functions.
- Added ``positive`` and ``negative`` functions.
- Added ``count_nonzero`` functions.
- Added ``concatenate`` functions.

**Fixed**

- ``nd.range`` now behaves properly when called as ``nd.range(x, null)`` (i.e. range from x to end).
- ``NDArray`` interpretation inside of Arrays would result in ``inhomogenous shape`` errors.
- Fixed ``NDArray.to_godot_array()`` producing garbage data and shapes.

Version 0.4 - 2024-10-03
------------------------
**Added**

- Added :ref:`NDRandomGenerator <class_NDRandomGenerator>`, created by ``nd.default_rng``. It offers ``.random()`` for floats, ``.integers`` for ints and ``.spawn()`` for child generators.
- Added new namespaces :ref:`ndb <class_ndb>`, :ref:`ndf <class_ndf>` and :ref:`ndi <class_ndi>`, for full tensor reductions to ``bool``, ``float`` and ``int``, respectively.
- Added ``nd.median``.
- ``NDArray`` is now iterable over the outermost dimension.
- ``NDArray`` conversion functions to and from ``Color``, ``Vector2``, ``Vector3``, ``Vector4``, ``Vector2i``, ``Vector3i``, ``Vector4i``, ``PackedVector2Array``, ``PackedVector3Array``, ``PackedVector4Array`` and ``PackedColorArray``.
- Added ``nd.as_array`` shorthands for every data type, e.g. ``nd.float32``.
- (Now really) added the ``logical_xor`` function.
- Added ``nd.eye``.
- Added ``nd.empty_like``, ``nd.full_like``, ``nd.ones_like`` and ``nd.zeros_like``.
- Added ``NDArray.strides()``, ``NDArray.strides_layout()``, and ``NDArray.strides_offset()``, through which you can inspect the strides properties of an ``NDArray`` / ``NDArray`` view.

**Changed**

- ``nd.array`` and ``nd.as_array``, ``NDArray.get_float``, ``NDArray.get_int``, ``NDArray.get_bool`` are now up to 2x faster.
- ``NDArray.to_godot_array`` now slices into the outermost dimension instead of flattening the array. To get floats and ints directly, use ``.to_packedxxx``.
- ``NDArray.to_packed_xxx`` now require 0D or 1D arrays to work. If the array is 2D, the conversion is not trivial, and a reshape should be used first.
- NumDot now uses ``Vector4i`` as a surrogate for range objects. They are represented as (bitmask, start, stop, step). This optimizes range creation, interpretation and memory use.

Version 0.3 - 2024-09-25
------------------------
**Added**

- Added the ``dot`` and ``reduce_dot`` functions.
- Added the ``matmul`` function.
- ``nd.array([...])`` can now handle more complex array inputs, e.g. an array of ``Vector2i``.
- Added the ``stack`` and ``unstack`` functions.
- Added :ref:`NDArray <class_NDArray>` ``to_bool`` and ``get_bool`` functions.
- ``nd.full`` now supports bools and arrays for the fill value.
- Axes, shape and permutation parameters now have support for more different argument types (including NDArrays).
- Added ``NUMDOT_COPY_FOR_ALL_INPLACE_OPERATIONS`` flag. This flag allows custom builds to de-optimize in-place operations even for optimal types. This reduces the binary size.
- Added ``NUMDOT_OPTIMIZE_ALL_INPLACE_OPERATIONS`` flag. This flag allows custom builds to optimize all in-place operations, even for non-optimal target types. This increases the binary size a lot and is not recommended.

**Changed**

- In-place operations with optimal destination types are now optimized by default.
- Removed ``NUMDOT_ASSIGN_INPLACE_DIRECTLY_INSTEAD_OF_COPYING_FIRST`` compile flag.

**Fixed**

- :ref:`NDArray <class_NDArray>` ``set`` didn't honor the index parameters, and didn't broadcast.

Version 0.2 - 2024-09-20
------------------------
**Added**

- Added an in-place API to :ref:`NDArray <class_NDArray>` objects, mirroring the :ref:`nd <class_nd>` API. In-place functions can substantially improve performance for small arrays, because creation of intermediate types is avoided.
- Added the ``NUMDOT_ASSIGN_INPLACE_DIRECTLY_INSTEAD_OF_COPYING_FIRST`` compiler flag, which improves performance of same-type assignment while increasing the binary size.
- Added the ``norm`` function (l0, l1, l2 and linf supported).
- Added the ``logical_xor`` function.
- Added the ``any`` and ``all`` functions.
- Added the ``square`` function.
- Added the ``clip`` function.
- ``nd.array`` can now interpret multi-dimensional boolean arrays.
- Documentation is now available in the editor.

**Changed**

- Reduced the binary size by half. In exchange, decrease performance of operations that need a cast before running by ~25%. The C define ``NUMDOT_CAST_INSTEAD_OF_COPY_FOR_ARGUMENTS`` lets you revert to the old behavior.
- Optimized the compiler arguments for the release binary. On web, it optimizes for size (~30% decrease). For downloadable binaries, it optimizes for performance (2% to 30% increase). You can use custom builds to change the default behavior.

**Fixed**

- Reduction functions now behave properly when casting (they used to crash or produce meaningless results).
- Array creation could often lead to the wrong dtype.
- ``nd.prod`` erroneously evaluated as ``nd.sum``.

Version 0.1 - 2024-09-17
------------------------
Initial release.
