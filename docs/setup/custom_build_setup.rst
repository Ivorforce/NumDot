.. _doc_custom_build_setup:

Custom Build Setup Guide
===========================

To build NumDot yourself, you will need basic proficiency with ``git`` and the command line. You will also need to have `Scons <https://scons.org>`_ installed.

Beyond that, your level of expertise defines the amount of things you can change: Some optimizations are very accessible, with no programming knowledge needed, while some may require you to dive into the code yourself.

Setup Considerations
--------------------

Godot is a multi-platform project. That means making custom builds involve building on multiple platforms. Usually, this is very difficult, but continuous integration (CI) a lot easier.

NumDot already has CI configured. To make use of it, you need fork the repository, make your changes, and trigger a build. To start, make an account on `GitHub <https://github.com>`_. Visit the `NumDot Repository <https://github.com/Ivorforce/NumDot>`_, and at the top, click "Fork".

Then, continue with the setup.

Setup
-----

First, clone the repository:

.. code-block:: bash

    git clone --recurse-submodules https://github.com/YourName/NumDot
    cd NumDot

To set up the codebase, you need to build godot-cpp once:

.. code-block:: bash

    cd godot-cpp
    # Replace platform with one of [macos, windows, linux]
    scons
    cd ..

Building NumDot locally
-----------------------

You should test your changes locally before submitting anything. The ``demo`` project exists for this reason. You can make a local build like so:

.. code-block:: bash

   # Replace platform with one of [macos, windows, linux]
   scons dev_build=yes install_dir=demo

**For macOS Users:** You should (currently) specify the target arch (``x86_64`` or ``arm64``) to avoid unnecessarily building a universal version, which takes twice as long.

.. code-block:: bash

   scons dev_build=yes install_dir=demo arch=x86_64

**For Windows Users:** It is recommended to use `MinGW <https://www.mingw-w64.org/>`__ to compile the project, as it more closely mimics Unix build semantics than MSVC. MinGW must be added to ``PATH``, and specified in the build command like so:

.. code-block:: bash

   scons dev_build=yes install_dir=demo use-mingw=yes

You should now be able to open the Demo project (``demo`` in the repository) in Godot, and test your build.

Making changes
--------------

You've set up your repository, and you've managed to test NumDot in the demo. Great! Now comes the most interesting part: Making changes. This is documented in :ref:`Custom Build Reference<doc_custom_build_reference>`

Making a cross-platform build
-----------------------------

With a fork, all you need to do now is trigger a build. Click on the ``Actions`` tab in your GitHub repository page (e.g. ``https://github.com/YourName/NumDot/actions``), and navigate to ``Build GDExtension``. Click the button ``Run workflow``, and wait for it to complete. Click the finished workflow, and download the zip file offered at the bottom. All you need to do now is extract its contents into your game project.

CLion Support
-------------

To get CLion support (for easier code editing), run this:

.. code-block:: bash

    scons compiledb=yes compile_commands.json
