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

In-Place Views
--------------

In the future, we want to support in-place modification without copying. This is not yet implemented.

.. code-block:: gdscript

    var packed := PackedFloat32Array([1, 2, 3])
    var a := nd.as_array(packed)
    a.add(a, 5)
    print(packed)  # [6, 7, 8]
