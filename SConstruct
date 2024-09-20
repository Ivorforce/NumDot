#!/usr/bin/env python
import os
import sys

from methods import print_error, print_warning

if not (os.path.isdir("godot-cpp") and os.listdir("godot-cpp")):
    print_error("""godot-cpp is not available within this folder, as Git submodules haven"t been initialized.
Run the following command to download godot-cpp:

    git submodule update --init --recursive""")
    sys.exit(1)

# ============================= Project Setup =============================

libname = "numdot"
projectdir = "demo"

env = Environment(tools=["default"], PLATFORM="")

# Load variables from custom.py, in case someone wants to store their own arguments.
customs = ["custom.py"]
customs = [os.path.abspath(path) for path in customs]
opts = Variables(customs, ARGUMENTS)

# ============================= Change defaults of godot-cpp =============================

# To read up on why exceptions should be enabled, read further below.
if ARGUMENTS.get("disable_exceptions", None):
    raise ValueError("NumDot does not currently support compiling without exceptions.")
ARGUMENTS["disable_exceptions"] = False

target = ARGUMENTS.get("target", "template_debug")
is_release = target == "template_release"

if ARGUMENTS.get("optimize", None) is None:
    # The default godot-cpp optimizes for speed
    if not is_release:
        # In dev, prioritize fast builds.
        # Godot-cpp defaults to optimizing speed (wat?).
        ARGUMENTS["optimize"] = "none"
    else:
        # On release, optimize by default.
        if ARGUMENTS["platform"] == "web":
            # For web, optimize binary size, can shrink by ~30%.
            ARGUMENTS["optimize"] = "size"
        else:
            # For download, optimize performance, can increase by 2% to 30%.
            ARGUMENTS["optimize"] = "speed"

# env["debug_symbols"] == False will strip debug symbols.
# It is False by default, unless dev_build is True.
# dev_build is a flag that should only be used by engine developers (supposedly).

# Load godot-cpp
env = SConscript("godot-cpp/SConstruct", {"env": env, "customs": customs})

# ============================= Change flags based on setup =============================

is_msvc = "is_msvc" in env and env["is_msvc"]
assert is_release == (env["target"] == "template_release")

# Web "requires target feature "simd128"", we should solve that but for now let"s just disable simd on web.
if env["platform"] not in ["web"]:
    env.Append(CCFLAGS=[
        # See https://xtensor.readthedocs.io/en/latest/build-options.html
        # See https://github.com/xtensor-stack/xsimd for supported list of simd extensions.
        # Choosing more will make your program faster, but also more incompatible to older machines.
        "-DXTENSOR_USE_XSIMD=1",
    ])

if env["platform"] == "windows":
    # At least the github runner needs bigobj to be enabled (otherwise it crashes).
    # is_msvc is set by godot-cpp.
    if is_msvc:
        env.Append(CCFLAGS=["/bigobj"])
    else:
        env.Append(CCFLAGS=["-Wa,-mbigobj"])

# TODO Figure out MSVC equivalents
if is_release:
    # Enable link-time optimization.
    # This further lets the compiler optimize, reduce binary size (~.5mb) or inline functions (possibly improving speeds).
    if is_msvc:
        env.Append(CCFLAGS=["/GL"])
        env.Append(LINKFLAGS=["/LTCG"])
    else:
        env.Append(CCFLAGS=["-flto"])
        env.Append(LINKFLAGS=["-flto"])

# You can also use "-march=native", which should enable all simd architectures your computer supports.
# Keep in mind the resulting binary will likely not work on many other computers.
#env.Append(CCFLAGS=["-march=native"])

# ============================= Actual source and lib setup =============================

env.Append(CPPPATH=["xtl/include", "xsimd/include", "xtensor/include"])

env.Append(CPPPATH=["src/"])
sources = Glob("src/*.cpp") + Glob("src/*/*.cpp")

if env["target"] in ["editor", "template_debug", "template_release"]:
    try:
        doc_data = env.GodotCPPDocData("src/gen/doc_data.gen.cpp", source=Glob("doc_classes/*.xml"))
        sources.append(doc_data)
    except AttributeError:
        print("Not including class reference as we're targeting a pre-4.3 baseline.")

lib_filename = "{}{}{}{}".format(env.subst("$SHLIBPREFIX"), libname, env["suffix"], env.subst("$SHLIBSUFFIX"))
lib_filepath = ""

if env["platform"] == "macos" or env["platform"] == "ios":
    # For signing, the dylibs need to be in a folder, along with the plist files.
    lib_filepath = "{}-{}.framework/".format(libname, env["platform"])

libraryfile = "build/addons/{}/{}/{}{}".format(libname, env["platform"], lib_filepath, lib_filename)
library = env.SharedLibrary(
    libraryfile,
    source=sources,
)

copy = env.Install("{}/addons/{}/{}/{}".format(projectdir, libname, env["platform"], lib_filepath), library),

default_args = [library, copy]
Default(*default_args)
