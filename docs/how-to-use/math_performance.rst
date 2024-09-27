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

<TODO>
