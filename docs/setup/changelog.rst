.. _doc_changelog:

Changelog
=========

Here you will find the release notes for each version of the library. Each section includes information about changes, improvements, and bug fixes.

.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

Upcoming Changes (master branch)
--------------------------------
**Added**

- Added the ``dot`` and ``reduce_dot`` functions.
- Added the ``matmul`` function.
- ``nd.array([...])`` can now handle more complex array inputs, e.g. an array of ``Vector2i``.
- Added the ``stack`` and ``unstack`` functions.
- Added :ref:`NDArray <class_NDArray>` ``to_bool`` and ``get_bool`` functions.
- ``nd.full`` now supports bools and arrays for the fill value.

**Fixed**

- :ref:`NDArray <class_NDArray>` ``set`` didn't honor the index parameters.

Version 0.2 - 2024-09-20
-----------------
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
