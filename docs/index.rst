:allow_comments: False

Introduction
============================

`NumDot <https://github.com/Ivorforce/NumDot>`__ is a tensor math and scientific computation library for the `Godot Engine <https://godotengine.org>`__.

NumDot provides a multidimensional array object (:ref:`NDArray <class_NDArray>`) and :ref:`many functions <class_nd>` for fast operations on arrays, including mathematical, logical, statistical, and more. It is inspired by the Python tensor math library, `NumPy <https://numpy.org>`__, and thus shares many semantics with it.

Motivation
==========

The NumDot library for Godot Engine is designed to make scientific and mathematical computations fast and easy using GDScript. Here are some reasons why you may want to consider using NumDot:

- **Performance**: NumDot is optimized for handling performance-intensive mathematical operations.
- **Convenience**: NumDot reduces the need for complex, hard to understand code by providing comprehensive mathematical functions.
- **Ease of Use**: User-friendly design simplifies complex mathematical tasks.
- **Consistency**: NumDot ensures accurate calculations consistent with NumPy.
- **Educational Benefit**: NumDot facilitates learning and teaching mathematical concepts within Godot.

In short, NumDot bridges the gap between high-level GDScript programming and efficient mathematical computations, enhancing both the development process and the final product.

If you need any help with NumDot, come by our `Discord Server <https://discord.gg/mwS2sW6V5M>`_ and have a chat.

Table of Contents
============================

.. Add :hidden: to each to hide them on this page. For now it's better to have them for quick navigation.

.. toctree::
   :maxdepth: 2
   :caption: Setup and Resources
   :name: sec-learn

   setup/how-to-install
   setup/changelog
   setup/custom_build_setup
   setup/custom_build_reference
   setup/alternatives

.. toctree::
   :maxdepth: 2
   :caption: How to Use
   :name: sec-learn

   how-to-use/getting_started
   how-to-use/tensors
   how-to-use/godot-interop
   how-to-use/numpy-xtensor-numdot
   how-to-use/math_performance

.. toctree::
   :maxdepth: 2
   :caption: Class Reference
   :name: sec-class-ref

   classes/class_nd
   classes/class_ndb
   classes/class_ndf
   classes/class_ndi
   classes/class_ndarray
   classes/class_ndrange
   classes/class_ndrandomgenerator
