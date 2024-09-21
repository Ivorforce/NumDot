.. _doc_custom_builds:

Custom Builds
=============

To build NumDot yourself, you will need basic proficiency with the command line. You will also need to have developer tools installed.

Beyond that, your level of expertise defines the amount of things you can change: Some optimizations are very accessible, with no programming knowledge needed, while some may require you to dive into the code yourself.

Setup Considerations
--------------------

Godot is a multi-platform project. That means making custom builds involve building on multiple platforms. Usually, this is very difficult, but Continuous Integration makes this *very easy* by comparison.

NumDot already has Continuous Integration configured. To make use of it, you just need fork the repository, and make your changes. This means that: To do that, make an account on `GitHub <https://github.com>`_. Visit the `NumDot Repository <https://github.com/Ivorforce/NumDot>`_, and at the top, click "Fork".

Then, continue with the setup.

Setup
-----

First, clone the repository:

.. code-block:: bash

    git clone https://github.com/YourName/NumDot
    cd NumDot

To set up the codebase, you need to build godot-cpp once:

.. code-block:: bash

    cd godot-cpp
    # Replace platform with one of [macos, windows, linux]
    scons platform=<platform> custom_api_file=../extension_api.json
    cd ..

Building NumDot locally
-----------------------

You should test your changes locally before submitting anything. The 'demo' project exists for this reason. You can make a local build like so:

.. code-block:: bash

   # Replace platform with one of [macos, windows, linux]
   scons platform=<platform> build_dir=demo target=template_debug

**For Windows Users:** It is recommended to use `MinGW <https://www.mingw-w64.org/>`__ to compile the project, as it more closely mimics Unix build semantics than MSVC. MinGW must be added to ``PATH``, and specified in the build command like so:

.. code-block:: bash

   scons platform=windows  build_dir=demo target=template_debug use-mingw=yes

You should now be able to open the Demo project (``demo`` in the repository) in Godot, and test your build.

Making changes
--------------

You've set up your repository, and you've managed to test NumDot in the demo. Great! Now comes the most interesting part: Making changes. This is documented in :ref:`Optimization Reference<doc_optimization_reference>`

Making a cross-platform build
-----------------------------

If you've forked the repository, all you need to do is tag a release:

.. code-block:: bash

    # Change <version> accordingly
    git tag release/custom/<version>
    git push --tags

The `GitHub Runner <https://github.com/Ivorforce/NumDot/blob/main/.github/workflows/build.yml>`__ will then make a build. It should complete within 10 minutes. On the ``Actions`` tab in your GitHub repository page (e.g. ``https://github.com/YourName/NumDot/actions``), you should see your release. Click it, and download the zip file offered at the bottom. All you need to do now is extract it into your game project.

CLion Support
-------------

To get CLion support (for easier code editing), run this:

.. code-block:: bash

    scons compiledb=yes compile_commands.json
