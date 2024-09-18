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

**Changed**

- Reduced the binary size by half. In exchange, decrease performance of operations that need a cast before running by ~25%. The C define ``NUMDOT_CAST_INSTEAD_OF_COPY_FOR_ARGUMENTS`` lets you revert to the old behavior.
- Add in-place assignment operators to :ref:`NDArray <class_NDArray>` objects, mirroring the :ref:`nd <class_nd>` API. These can substantially improve performance for small arrays, because creation of intermediate types is avoided.

Version 0.1 - 2024-09-17
------------------------
Initial release.
