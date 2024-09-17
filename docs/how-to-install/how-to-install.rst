.. _doc_how_to_install:

How to Install
==============

Asset Library (recommended)
---------------------------

The easiest way to install NumDot is through the asset library. This is not yet supported (but will be very soon!).

Manual Download
---------------

You can manually download the `latest release <https://github.com/Ivorforce/NumDot/releases>`__ from GitHub.
To install, simply place the files into your Godot project.

.. _doc_how_to_install_manual_build:
Manual Build
------------

You may want to manually build NumDot, for example to:

- **Reduce Build Size:** The default NumDot build is fairly large. You can compile it with fewer features to reduce the binary size.
- **Adjust Optimization Options:** You can accelerate your mathematical operations further, by sacrificing compatibility and increasing binary size.
- **Custom Extension:** You can extend NumDot's functionality by interfacing with `xtensor <http://xtensor.readthedocs.io>`__ directly, substantially improving performance of individual computations.

The steps to setting up the workspace are outlined in NumDot manually are outlined in `Contributing.md <https://github.com/Ivorforce/NumDot/blob/main/CONTRIBUTING.md>`__.

NumDot releases are built by a `GitHub workflow <https://github.com/Ivorforce/NumDot/blob/main/.github/workflows/build.yml>`__. An easy way to get a custom build is to `fork NumDot <https://github.com/Ivorforce/NumDot/>`__ and triggering the build by tagging a release.

Alternatively, you can build NumDot locally. To do this, invoke the scons build with all needed platforms to create the needed binaries.
