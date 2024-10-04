.. _doc_custom_build_reference:

Custom Build Reference
======================

This article describes the ways you can change the NumDot to optimize it differently. If you're here to make a :ref:`custom build<doc_custom_build_setup>`, please read that article first.

Scons Options
-------------

The most accessible way to re-optimize NumDot is through the options offered by NumDot itself, through scons. To get a list of all available options, run:

.. code-block:: bash

    scons --help

That being said, here is some information about the most interesting options:

- ``optimize``, one of [``size`` ``speed`` ``speed_trace`` ``none``]:

    - Sets compiler flags ``-Os``, ``-O3`` or ``-O2``, respectively. You can achieve up to 50% decrease in binary size, or increase of performance, with different values. ``speed_trace`` is a slightly weaker version of ``speed`` and not recommended.

- ``arch``:

    - Builds for one specific architecture. This can be relevant on macOS, where arch defaults to 'universal'. You can achieve lower binary sizes by building every arch separately.

- ``optimize_for_arch``:

    - Optimize the build for one specific (sub-)architecture. This makes the build incompatible with machines of different architectures, but can improve performance substantially.

    - Use ``optimize_for_arch=native`` to automatically select the architecture of the machine you're building on.

    - Otherwise, look up arches on the gcc docs, e.g. `x86_64 arches <https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html>`_ or `arm64 arches <https://gcc.gnu.org/onlinedocs/gcc/AArch64-Options.html>`_.

- ``use_xsimd``, one of [``yes`` ``no`` ``auto``]:

    - Whether to use `xsimd <https://xsimd.readthedocs.io/en/latest/>`_ to accelerate contiguous memory operations. Defaults to 'no' on web (unsupported as of yet) and yes elsewhere.

- ``define=NUMDOT_COPY_FOR_ALL_INPLACE_OPERATIONS``

    - Don't optimize in-place operations (e.g. ``array.assign_add(a, b)``, even those of optimal type. This reduces their performance, but reduces the binary size.

- ``define=NUMDOT_OPTIMIZE_ALL_INPLACE_OPERATIONS``

    - Optimize all in-place operations (e.g. ``array.assign_add(a, b)``, even those of non-optimal type. This improves the performance of these inplace non-optimal type operations, but increases the binary size by a lot.

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


Disabling Features
------------------

One of the easiest ways to save space is to disable features you don't need. Any disabled functions will error when they are called, instead of evaluating. To disable a feature, pass its macro into a ``define`` argument, like above, e.g.:

``define=NUMDOT_DISABLE_ALL_OPTIONAL_FUNCTIONS``

**Toggleable features:**

- ``NUMDOT_DISABLE_ALL_OPTIONAL_FUNCTIONS``: Same as passing all flags below individually. Brings the binary size to ~15% (~1.5mb).
- ``NUMDOT_DISABLE_ALLOCATION_FUNCTIONS``
- ``NUMDOT_DISABLE_COMPARISON_FUNCTIONS``
- ``NUMDOT_DISABLE_LOGICAL_FUNCTIONS``
- ``NUMDOT_DISABLE_REDUCTION_FUNCTIONS``
- ``NUMDOT_DISABLE_ROUNDING_FUNCTIONS``
- ``NUMDOT_DISABLE_TRIGONOMETRY_FUNCTIONS``
- ``NUMDOT_DISABLE_MATH_FUNCTIONS``
- ``NUMDOT_DISABLE_RANDOM_FUNCTIONS``
- ``NUMDOT_DISABLE_GODOT_CONVERSION_FUNCTIONS``

You can also pass ``NUMDOT_DISABLE_ALL_OPTIONAL_FUNCTIONS`` to disable all functions, and only enable those you need. Enable macros have precedence over disable macros.

- ``NUMDOT_ENABLE_ALLOCATION_FUNCTIONS``
- ``NUMDOT_ENABLE_COMPARISON_FUNCTIONS``
- ``NUMDOT_ENABLE_LOGICAL_FUNCTIONS``
- ``NUMDOT_ENABLE_REDUCTION_FUNCTIONS``
- ``NUMDOT_ENABLE_ROUNDING_FUNCTIONS``
- ``NUMDOT_ENABLE_TRIGONOMETRY_FUNCTIONS``
- ``NUMDOT_ENABLE_MATH_FUNCTIONS``
- ``NUMDOT_ENABLE_RANDOM_FUNCTIONS``
- ``NUMDOT_ENABLE_GODOT_CONVERSION_FUNCTIONS``


Editing Code
------------

The most powerful way to get more out of NumDot is to edit its code. For example, you could add functions that interface with ``xtensor`` directly, performing a specific operation you need. This operation will be both extremely fast *and* have a small binary size, so in some cases it may be worth it to go this far.

You'll need decent knowledge of C++ to make it work. See `Contributing.md <https://github.com/Ivorforce/NumDot/blob/main/CONTRIBUTING.md>`_ for a short introduction into its architecture.

If you need any help, we're happy to assist. Come by our `Discord Server <https://discord.gg/ch5MAsgKxF>`_ and have a chat.
