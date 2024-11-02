.. _doc_godot_interop:

Godot Interoperability
======================

Explicit Adaptations
--------------------

NumDot can adapt most `Variant <https://docs.godotengine.org/en/stable/classes/class_variant.html>`__ objects implicitly. You can explicitly convert back to godot types:

.. code-block:: gdscript

    var packed := PackedFloat32Array([1, 2, 3])
    var a := nd.add(packed, 5)
    packed = a.to_packed_float32_array()
    print(packed)  # [6, 7, 8]

ND-Arrays produced from packed arrays are special, in that they can produce instantaneous copy-on-write copies of the same type. In the above code, the latter assignment to packed happens instantaneously, because ``a`` is backed by a ``PackedFloat32Array``.

In-Place Views
--------------

You can also directly assign to an array, rather than a new one. This can be faster than creating a new array, because memory can be re-used.

.. code-block:: gdscript

    var packed := PackedFloat32Array([1, 2, 3])
    var a := nd.as_array(packed)
    a.assign_add(a, 5)
    a.assign_multiply(a, 2)
    print(packed)  # [6, 7, 8]

Godot-Native Reductions
-----------------------

When you're performing no-axis reductions, and are planning to use the results in scalar computation, you can use the appropriate APIs (:ref:`ndb <class_ndb>`, :ref:`ndf <class_ndf>` and :ref:`ndi <class_ndi>`).

.. code-block:: gdscript

    if ndb.all(tensor):
        print("All")

    if ndf.mean(tensor) > 0.2:
        print("mean > 0.2")

    if ndi.sum(tensor) > 10:
        print("sum > 10")
