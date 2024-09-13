#!/usr/bin/env python
import os
import sys
from SCons.Script import Variables, Command, File, DefaultEnvironment

# This works like passing disable_exceptions=false by default.
# To read up on why exceptions must be enabled, read further below.
env = Environment()
env['disable_exceptions'] = False

env = SConscript("godot-cpp/SConstruct", 'env')

# For reference:
# - CCFLAGS are compilation flags shared between C and C++
# - CFLAGS are for C-specific compilation flags
# - CXXFLAGS are for C++-specific compilation flags
# - CPPFLAGS are for pre-processor flags
# - CPPDEFINES are for pre-processor defines
# - LINKFLAGS are for linking flags

env.Append(CPPFLAGS=[
    '-DXTENSOR_USE_XSIMD=1',
    
    # Explicitly enable exceptions (see https://github.com/godotengine/godot-cpp/blob/master/CMakeLists.txt).
    # '-DGODOT_DISABLE_EXCEPTIONS=OFF',
    # XTensor could disable exceptions, but then we would have to duplicate all our checks.
    # Would have to be passed to xtensor build too, I think.
    # '-DXTENSOR_DISABLE_EXCEPTIONS=1',
    
    # ffast-math: See https://stackoverflow.com/questions/57442255/xtensor-and-xsimd-improve-performance-on-reduction
    # And https://stackoverflow.com/questions/7420665/what-does-gccs-ffast-math-actually-do
    # We should not enable this flag by default, because infinite maths is definitely expected in many situations.
    # '-ffast-math',
    # See https://github.com/xtensor-stack/xsimd for supported list of simd.
    # Choosing more will make your program faster, but also more incompatible to older machines.
    '-msse2', '-msse3', '-msse4.1', '-msse4.2', '-mavx'
])

# You can also use '-march=native' instead, which will enable everything your computer has.
# Keep in mind the resulting binary will likely not work on many other computers.
#env.Append(CPPFLAGS=['-DXTENSOR_USE_XSIMD=1', '-march=native'])

env.Append(CPPPATH=["xtl/include", "xsimd/include", "xtensor/include"])

env.Append(CPPPATH=["src/"])
sources = Glob("src/*.cpp") + Glob("src/*/*.cpp")

if env["platform"] == "macos":
    target_file_path = "demo/bin/libnumdot.{}.{}.framework/libnumdot.{}.{}".format(
        env["platform"], env["target"], env["platform"], env["target"]
    )
else:
    target_file_path = "demo/bin/libnumdot{}{}".format(env["suffix"], env["SHLIBSUFFIX"])

# Via https://docs.godotengine.org/en/stable/tutorials/scripting/gdextension/gdextension_docs_system.html
if env["target"] in ["editor", "template_debug"]:
    try:
        doc_data = env.GodotCPPDocData("src/gen/doc_data.gen.cpp", source=Glob("doc_classes/*.xml"))
        sources.append(doc_data)
    except AttributeError:
        print("Not including class reference as we're targeting a pre-4.3 baseline.")

library = env.SharedLibrary(
    target_file_path,
    source=sources,
)

Default(library)
