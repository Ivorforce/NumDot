.. _doc_alternatives:

Alternatives to NumDot
======================

If you want to use fast tensor math in godot, there are alternative approaches you can take. This article lists your options, and describes the differences.

NumDot
------

This library: A NumPy-like tensor math library made for Godot.

**Pro**

- Good integration in the Godot ecosystem.
- Does not require any additional language beyond gdscript (or C#).
- Easy to install (through the asset library).
- Accessible documentation.
- Developed for games, with according priorities.
- (Ability to tweak binary size, WIP).
- (Ability to run operations in-place on godot types, WIP).

**Con**

- No ecosystem, small userbase.
- Young library with many uncertainties.
- Important parts of NumPy not yet covered.

NumPy
-----

`NumPy <https://numpy.org>`_ is the most popular tensor math solution.

**Pro**

- Largest Ecosystem of all solutions.
- Very Mature.
- Large existing userbase.
- Ability to prepare code in ipython notebooks.

**Con**

- Requires another language as gdextension (python), increasing complexity.
- Requires the full python interpreter, and the large NumPy binary (150+mb).
- No interoperability with Godot types.

Note: You may want to consider `TensorFlow <https://www.tensorflow.org>`_ for extreme projects, which can run on the GPU. Getting it to run can be difficult though.

xtensor
-------

`xtensor <https://github.com/xtensor-stack/xtensor>`_ is a numpy-like library made for C++ developers.

NumDot uses xtensor under the hood. If you want to use xtensor, consider forking or extending NumDot instead, using a :ref:`manual build<doc_how_to_install_manual_build>`.

**Pro**

- Extremely Fast, SIMD accelerated.
- Good coverage of NumPy API.
- Header only, i.e. very small binary size.

**Con**

- Requires another language as gdextension (C++), increasing complexity.
- Requires using a low level language (C++), which can be difficult.
- Fairly small ecosystem.
- No interoperability with Godot types.

NumSharp
-------

`NumSharp <https://github.com/SciSharp/NumSharp>`_ is a NumPy port for the C# ecosystem.

**Pro**

- Well integrated in the C# ecosystem.
- No gdextension needed.
- Reasonable binary size (~20mb).
- Good coverage of NumPy API.

**Con**

- Cannot use this API from gdscript.
- No built-in interoperability with Godot types.
- Some performance features missing (though it may be fast enough for your needs):

    - No in-place operations, slowing down repeated computation due to repeated allocation.

    - No SIMD acceleration, implementation is native C#.

- Fairly small ecosystem.
- No online documentation (granted it's very close to the NumPy API).
- No way to reduce binary size.
