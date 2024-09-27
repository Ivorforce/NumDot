.. _doc_math_performance:

Optimizing Performance of Operations
====================================

If you find your mathematical operation is running too slow, consider these steps to optimize:

1) **Question your Algorithm:** Are you using the optimal algorithm for the task? Perhaps it is possible to:

    - Use an algorithm with better `runtime complexity <https://en.wikipedia.org/wiki/Time_complexity>`_ (e.g. ``O(n log n)`` instead of ``O(n^2)``.

    - Avoid slow functions. A famous example of such an optimization is the `fast inverse square root <https://en.wikipedia.org/wiki/Fast_inverse_square_root>`_.

2) **Optimize NumDot Use:** You may be able to speed up your algorithm by using specific tricks to speed up your algorithm, documented in this article.

3) **Custom Build:** When you're sure you optimized everything you can, you can substantially speed up your algorithm by implementing it in C++, interfacing with ``xtensor`` directly. This is documented in the articles for :ref:`Custom Builds<_doc_custom_build_setup>`.

Optimizing NumDot Use
---------------------

Vectorization
^^^^^^^^^^^^^

The most common mistake when using tensor math libraries is to not vectorize enough. This means using manual iteration, when a broadcasting iteration could be used.

Consider the following example:

.. code-block:: gdscript

    var vectors := nd.zeros([1000, 2])
    for i in vectors.shape()[0]:
        var prog := i / 10.0
        vectors.set(Vector2(sin(prog), cos(prog)), i)

With vectorization, it would execute much, much faster:

.. code-block:: gdscript

    var vectors := nd.stack([1000, 2])
    var prog := nd.divide(nd.arange(1000), 10)
    vectors.set(nd.sin(prog), null, 0)
    vectors.set(nd.cos(prog), null, 1)

As a rule of thumb: The fewer calls you make to NumDot, the faster your algorithm executes.


In-Place Operations
^^^^^^^^^^^^^^^^^^^

Every operation in :ref:`nd <class_nd>` allocates new memory. Avoiding new allocations, especially for repeated operations, can improve performance of your operations by up to 2x.

Consider this example:

.. code-block:: gdscript

    var positions: NDArray
    var velocities: NDArray

    func _ready():
        # TODO Use random when we have it
        positions = nd.zeros([1000, 2])
        velocities = nd.ones([1000, 2])

    func _update():
        positions = nd.add(positions, velocities)

It would be much faster to directly assign to ``positions`` using in-place operations:

.. code-block:: gdscript

    # [...]

    func _update():
        positions.assign_add(positions, velocities)

Godot Conversions
^^^^^^^^^^^^^^^^^

:ref:`NDArray <class_NDArray>` has accelerated functions for godot ``Variant`` types:

.. code-block:: gdscript

    # Slow: This access creates an intermediate 0-D tensor.
    var f: float = array.get(5).to_float()

    # Fast: This access directly returns a float.
    var f: float = array.get_float(5)

.. code-block:: gdscript

    # Slow: Conversion is not accelerated.
    var packed := PackedFloat32Array()
    packed.resize(array.size())
    for i in array.shape()[0]:
        packed[i] = array.gef_float(i)

    # Fast: Conversion is accelerated.
    var packed := array.to_float32_array()

In the future, NumDot will have an API for accelerated full-tensor reductions (like ``ndf.mean(array)``).
