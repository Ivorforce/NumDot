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

- ``nd.outer`` and ``nd.inner`` functions for dedicated vector multiplication.
- ``nd.squeeze`` function.
- Mathematical constants (``pi``, ``e``, ``euler_gamma``, ``inf``, ``nan``). These are currently added as functions, due to limitations of Godot's APIs.

**Changed**

- ``nd.reshape`` no longer re-interprets the previous shape (if layout is not row-major). Instead, it iterates the previous array in the correct order, filling elements one by one.
- ``nd.flatten`` no longer makes a copy if it doesn't need to.

Version 0.9 - 2025-04-29
------------------------

**Changed**

- Update ``xtensor`` to ``0.26.0``, ``xsimd`` to ``13.2.0``, ``xtl`` to ``0.8.0``. This may come with both improvements and regressions.

**Fixed**

- ``nd.any`` and ``nd.all`` always incorrectly evaluated to ``true`` and ``false`` respectively.
- Fix ``nd.linspace`` for non-floating types.


Version 0.8 - 2025-03-05
------------------------
The backend of NumDot has been completely rewritten! The changes shrink the binary size in half, and make some functions a lot faster. Some bugs may have been silently fixed or introduced in the process. `Let us know <https://github.com/Ivorforce/NumDot/issues>`_ if you run into any trouble!
In addition, there's unit tests now that check validity of functions against NumPy implementations as the ground truth. This should help make NumDot functions more reliable.

**Added**

- Added ``nd.load`` and ``nd.saveb`` functions, to read and write files from ``.npy``.
- Added ``array_equiv`` function.
- Added conversion functions for ``Plane``, ``Quaternion``, ``Projection`` and ``Basis``.
- Added ``NDArray`` functions for getting slices as variant types (if shape is compatible).
- Added ``complex64`` and ``complex128`` array creation shorthand functions.
- NumDot now appears in the ``plugins`` section of the editor preferences.
- NumDot now builds for Android (``x86_32``, ``x86_64``, ``arm32``).

**Changed**

- Optimizations in the build scheme and code architecture reduced the binary size by 50%.
- ``array_equal`` now also checks for shape equivalence, and doesn't fail if the shapes are not broadcastable.
- 1-D array assignment is now about 3.5 times faster.
- Single-slice indexing now has a lower latency.
- Contiguous array conversions (from ``NDArray`` to godot arrays) has been optimized, and can be several times faster.
- ``nd.add`` and ``nd.abs``, ``nd.remainder``, ``nd.pow`` and ``nd.remainder`` no longer promote values to higher bit counts.
- ``complex`` numbers can now be booleanized in some situations.

**Fixed**

- Various functions resulting in bools were broken. This is now fixed.
- ``NDArray`` found in godot arrays will now properly type hint the resulting array, avoiding accidental promotion.
- ``arange`` was producing garbage data.
- ``bitwise_left_shift`` and ``bitwise_right_shift`` were incorrectly promoting, and producing undefined behavior when the shift was larger than the bit count (now it defaults to 0).

Version 0.7 - 2024-11-12
------------------------
**Added**

- Added complex numbers data types (``complex64`` and ``complex128``).
- Added ``real``, ``imag``, ``conjugate`` and ``angle`` functions for complex numbers.
- Added ``complex_as_vector`` and ``vector_as_complex`` functions for convenient complex number creation and manipulation, similar to ``real`` and ``imag``.
- Added ``any`` layout type, which may bring tiny speed improvements.
- Added ``fft`` and ``fft_freq`` functions.
- Added ``pad`` function.
- Added ``cross`` function.
- Added ``ndarray.buffer_size`` and ``ndarray.buffer_dtype`` functions for investigation of underlying buffer types.
- Added bitwise functions (``bitwise_and``, ``bitwise_or``, ``bitwise_xor``, ``bitwise_not``, ``bitwise_left_shift``, ``bitwise_right_shift``).
- Added matrix ``diagonal``, ``diag`` and ``trace`` functions.
- Added ``transpose`` and ``flatten`` to ``NDArray`` methods.
- Added ``is_close``, ``array_equal`` and ``all_close`` functions.
- Added ``is_nan``, ``is_inf`` and ``is_finite`` functions.

**Changed**

- In-place adaptations of native godot types speed up conversions (to and from NumDot). In particular, in-place adaptations of packed arrays do not need to copy data on read, and will produce instantaneous copy-on-write copies on ``to_packed_xxx_array`` calls for the same type.
- ``ndarray.array_size_in_bytes`` is now called ``ndarray.buffer_size_in_bytes``.
- Custom builds can now disable each function / feature individually. This allows for very fine control of what to include in a custom build, which can reduce NumDot builds down to almost 0mb.
- Removed ``NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS`` to improve usability. A similar option may be re-added to de-optimize conversions to save space.
- Functions no longer declare unnecessary default values.
- ``transpose()`` can now be called without passing a parameter, which reverses the axes.

**Fixed**

- ``arange`` produced 0-size arrays when at least two arguments were passed.

Version 0.6 - 2024-10-28
------------------------
**Added**

- ``randn`` function (random sampling from a normal distribution)
- For custom builds, OpenMP support (through the new ``openmp_threshold`` compile option, disabled by default). This requires your compiler to support OpenMP.
- Add support for web exports (wasm32).

**Changed**

- Contiguous scalar assignment (e.g. ``array.set(0)``) is now about 20x as fast as before.
- Mask assignments with scalars are now a bit faster.

**Fixed**

- Assignment from boolean to boolean arrays didn't work properly.
- Setting with a mask of incompatible shape to the array didn't properly fail.

Version 0.5 - 2024-10-07
------------------------
**Added**

- Bounds checks are now enabled everywhere.
- Negative indices are now supported everywhere.
- Added boolean mask indexing, e.g. ``a.set(5, nd.greater(a, 5))``.
- Added index list indexing, e.g. ``a.set(5, [[0, 1], [4, 2]])``.
- Added a basic ``convolve`` function.
- Added the ``sliding_window_view`` function.
- Added ``array.copy()`` and ``nd.copy(array)`` functions.
- Added ``positive`` and ``negative`` functions.
- Added ``count_nonzero`` functions.
- Added ``concatenate``, ``hstack`` and ``vstack`` functions.
- Added ``split``, ``hsplit`` and ``vsplit`` functions.
- Added the ``tile`` function.
- Added scalar optimizations for binary functions. This will greatly accelerate calls like ``nd.add(array, 5)``, at the cost of some binary size. This behavior can be disabled with the build flag ``NUMDOT_DISABLE_SCALAR_OPTIMIZATION``.
- ``nd.matmul`` can now handle matrix-vector multiplication.

**Changed**

- Added ``array.set(x)`` should now be slightly faster when only a single element is updated.
- Accelerated ``reduce_dot``. This also affects ``matmul``, ``dot``, and ``convolve`` operations.

**Fixed**

- ``nd.range`` now behaves properly when called as ``nd.range(x, null)`` (i.e. range from x to end).
- ``NDArray`` interpretation inside of Arrays would result in ``inhomogenous shape`` errors.
- Fixed ``NDArray.to_godot_array()`` producing garbage data and shapes.
- Fixed ``NDArray.to_packed_xxx`` producing arrays that were too large.
- Fixed ``zeros_like`` and similar producing garbage arrays when the dtype is not given.

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
