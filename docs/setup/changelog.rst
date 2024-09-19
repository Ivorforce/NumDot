.. _doc_changelog:

Changelog
=========

Here you will find the release notes for each version of the library. Each section includes information about changes, improvements, and bug fixes.

.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

Version 0.2 - 2024-09-19
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

**Changed**

- Reduced the binary size by half. In exchange, decrease performance of operations that need a cast before running by ~25%. The C define ``NUMDOT_CAST_INSTEAD_OF_COPY_FOR_ARGUMENTS`` lets you revert to the old behavior.

**Fixed**

- Reduction functions now behave properly when casting (they used to crash or produce meaningless results).
- Array creation could often lead to the wrong dtype.
- ``nd.prod`` erroneously evaluated as ``nd.sum``.

Version 0.1 - 2024-09-17
------------------------
Initial release.
