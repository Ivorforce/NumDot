.. _doc_optimization_reference:

Optimization Reference
======================

This article describes the ways you can change the NumDot to optimize it differently. If you're here to make a :ref:`custom build<doc_custom_builds>`, please read that article first.

Scons Options
-------------

The most accessible way to re-optimize NumDot is through the options offered by NumDot itself, through scons. To get a list of all available options, run:

.. code-block:: bash

    scons --help

That being said, here is some information about the most interesting options:

- ``optimize`` (``size``, ``speed`` or ``speed_trace``):

    - Sets compiler flags ``-Os``, ``-O3`` or ``-O2``, respectively. You can achieve up to 50% decrease in binary size, or increase of performance, with different values. ``speed_trace`` is a slightly weaker version of ``speed`` and not recommended.

- ``define=NUMDOT_ASSIGN_INPLACE_DIRECTLY_INSTEAD_OF_COPYING_FIRST``

    - Optimize in-place operations (e.g. ``array.assign_add(a, b)``. This substantially improves their performance, but can also increase the binary size but up to 100%.

- ``define=NUMDOT_CAST_INSTEAD_OF_COPY_FOR_ARGUMENTS``

    - Optimize wrong-type argument conversion (e.g. ``nd.sqrt``, which promotes int arguments to ``float64``). The argument improves performance of cross datatype conversions, but also increases binary size.

**Note:** You can have as many ``define=[...]`` arguments as you wish.

You can test building with these options locally. To get them to be permanent, edit the SConstruct file, and add your needed changes at the spot intended for it:

.. code-block:: python

    # CUSTOM BUILD FLAGS
    # Add your build flags here:
    if is_release:
        ARGUMENTS["optimize"] = "speed"  # For normal flags, like optimize
        env.Append(CPPDEFINES=["NUMDOT_XXX"])  # For all C macros (``define=``).


Editing Code
------------

The most powerful way to get more out of NumDot is to edit its code.

You're on your own here, but you'll need decent knowledge of C++ to make it work. See `Contributing.md <https://github.com/Ivorforce/NumDot/blob/main/CONTRIBUTING.md>`_ for a short introduction into its architecture.
