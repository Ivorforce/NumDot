.. _doc_changelog:

Changelog
=========

Here you will find the release notes for each version of the library. Each section includes information about changes, improvements, and bug fixes.

.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

Latest (Unstable)
-----------------
**Added**

- Added an in-place to API :ref:`NDArray <class_NDArray>` objects, mirroring the :ref:`nd <class_nd>` API. In-place functions can substantially improve performance for small arrays, because creation of intermediate types is avoided.
- Added the ``NUMDOT_ASSIGN_INPLACE_DIRECTLY_INSTEAD_OF_COPYING_FIRST`` compiler flag, which improves performance of same-type assignment while increasing the binary size.

**Changed**

- Reduced the binary size by half. In exchange, decrease performance of operations that need a cast before running by ~25%. The C define ``NUMDOT_CAST_INSTEAD_OF_COPY_FOR_ARGUMENTS`` lets you revert to the old behavior.

Version 0.1 - 2024-09-17
------------------------
Initial release.
