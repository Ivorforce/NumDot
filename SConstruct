#!/usr/bin/env python
import os
import sys
from SCons.Script import Command, File, DefaultEnvironment

def subcommands(env, *, target_files, build_dir, commands):
    # Create the build dir
    os.makedirs(build_dir, exist_ok=True)

    actions = [
        f'cd {build_dir} && {command}'
        for command in commands
    ]

    # Create placeholder files representing the build target files
    targets = [File(target_file) for target_file in target_files]
    
    env.Command(
        target=targets,
        source=[],
        action=actions,
    )

env = SConscript("godot-cpp/SConstruct")

# For reference:
# - CCFLAGS are compilation flags shared between C and C++
# - CFLAGS are for C-specific compilation flags
# - CXXFLAGS are for C++-specific compilation flags
# - CPPFLAGS are for pre-processor flags
# - CPPDEFINES are for pre-processor defines
# - LINKFLAGS are for linking flags

# xtl and xtensor are header only, but we might as well use their scripts so everything coheres.
subcommands(
    env,
    target_files=['xtl/build/include/xtl/xtl.hpp'],
    build_dir='xtl/build',
    commands=[
        'cmake ../',
        'cmake --install . --prefix .'
    ],
)
subcommands(
    env,
    target_files=['xsimd/build/include/xsimd/xsimd.hpp'],
    build_dir='xsimd/build',
    commands=[
        "cmake ../ -DENABLE_XTL_COMPLEX=1 -Dxtl_DIR='../xtl/build/'",
        'cmake --install . --prefix .'
    ],
)
subcommands(
    env,
    target_files=['xtensor/build/include/xtensor/xtensor.hpp'],
    build_dir='xtensor/build',
    commands=[
        # Exceptions are disabled in godot in general.
        "cmake ../ -DXTENSOR_USE_XSIMD=1 -Dxtl_DIR='../xtl/build/' -Dxsimd_DIR='../xsimd/build/'",
        'cmake --install . --prefix .'
    ],
)


env.Append(CPPFLAGS=[
    '-DXTENSOR_USE_XSIMD=1',
    
    # Explicitly enable exceptions (see https://github.com/godotengine/godot-cpp/blob/master/CMakeLists.txt).
    # '-DGODOT_DISABLE_EXCEPTIONS=OFF',
    # XTensor could disable exceptions, but then we would have to duplicate all our checks.
    # Would have to be passed to xtensor build too, I think.
    # '-DXTENSOR_DISABLE_EXCEPTIONS=1',
    
    # ffast-math: See https://stackoverflow.com/questions/57442255/xtensor-and-xsimd-improve-performance-on-reduction
    # And https://stackoverflow.com/questions/7420665/what-does-gccs-ffast-math-actually-do
    # TODO Should we have non-finite math?
    '-ffast-math',
    # See https://github.com/xtensor-stack/xsimd for supported list of simd.
    # Choosing more will make your program faster, but also more incompatible to older machines.
    '-msse2', '-msse3', '-msse4.1', '-msse4.2', '-mavx'
])

# You can also use '-march=native' instead, which will enable everything your computer has.
# Keep in mind the resulting binary will likely not work on many other computers.
#env.Append(CPPFLAGS=['-DXTENSOR_USE_XSIMD=1', '-march=native'])

env.Append(CPPPATH=["xtl/build/include", "xsimd/build/include", "xtensor/build/include"])

env.Append(CPPPATH=["src/"])
sources = Glob("src/*.cpp")

if env["platform"] == "macos":
    target_file_path = "demo/bin/libnumdot.{}.{}.framework/libnumdot.{}.{}".format(
        env["platform"], env["target"], env["platform"], env["target"]
    )
else:
    target_file_path = "demo/bin/libnumdot{}{}".format(env["suffix"], env["SHLIBSUFFIX"])

env.Depends('xsimd/build/include/xsimd/xsimd.hpp', ['xtl/build/include/xtl/xtl.hpp'])
env.Depends('xtensor/build/include/xtensor/xtensor.hpp', ['xtl/build/include/xtl/xtl.hpp', 'xsimd/build/include/xsimd/xsimd.hpp'])
env.Depends(target_file_path, ['xtensor/build/include/xtensor/xtensor.hpp'])

library = env.SharedLibrary(
    target_file_path,
    source=sources,
)

Default(library)
