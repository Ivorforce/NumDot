#!/usr/bin/env python
import os
import sys
from SCons.Variables.BoolVariable import _text2bool

from methods import print_error, print_warning

if not (os.path.isdir("godot-cpp") and os.listdir("godot-cpp")):
    print_error("""godot-cpp is not available within this folder, as Git submodules haven"t been initialized.
Run the following command to download godot-cpp:

    git submodule update --init --recursive""")
    sys.exit(1)

# ============================= Project Setup =============================

libname = "numdot"

# Allow additional defines, see https://scons.org/doc/production/HTML/scons-user/ch10s02.html
cppdefines = []
for key, value in ARGLIST:
    if key == 'define':
        cppdefines.append(value)

env = Environment(tools=["default"], PLATFORM="", CPPDEFINES=cppdefines)

# Load variables from custom.py, in case someone wants to store their own arguments.
customs = ["custom.py"]
customs = [os.path.abspath(path) for path in customs]
opts = Variables(customs, ARGUMENTS)
opts.Add(
    PathVariable(
        key="build_dir",
        help="Target location of the binary (./build by default)",
        default=env.get("build_dir", "build"),
    )
)
opts.Add(
    "use_xsimd",
    "Whether to use xsimd, accelerating contiguous memory computation. Defaults to no on web and yes elsewhere.",
    "auto"
)
opts.Add(
    "optimize_for_arch",
    "Enable all optimizations the arch supports, making the build incompatible with other machines. Use 'native' to optimize for this machine. Note that on macOS, setting this option also requires setting arch= to a specific arch, e.g. arch=x86_64 or arch=arm64.",
    "",
)
opts.Update(env)

use_xsimd = env["use_xsimd"]
if ARGUMENTS.get("use_xsimd", "auto") == "auto":
    # Web "requires target feature "simd128"", we should solve that but for now let"s just disable simd on web.
    use_xsimd = ARGUMENTS.get("platform", None) != "web"
else:
    use_xsimd = _text2bool(use_xsimd)

optimize_for_arch = env["optimize_for_arch"]

# TODO If we don't delete our own arguments, the godot-cpp SConscript will complain.
# There must be a better way?
ARGUMENTS.pop("build_dir", None)
ARGUMENTS.pop("define", None)
ARGUMENTS.pop("use_xsimd", None)
ARGUMENTS.pop("optimize_for_this_machine", None)

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

if optimize_for_arch:
    # Yo-march improves performance, makes the build incompatible with most other machines.
    env.Append(CPPFLAGS=[f"-march={optimize_for_arch}"])

# CUSTOM BUILD FLAGS
# Add your build flags here:
# if is_release:
#     ARGUMENTS["optimize"] = "speed"  # For normal flags, like optimize
#     env.Append(CPPDEFINES=["NUMDOT_XXX"])  # For all C macros (``define=``).

# Load godot-cpp
env = SConscript("godot-cpp/SConstruct", {"env": env, "customs": customs})

# ============================= Change flags based on setup =============================

is_msvc = "is_msvc" in env and env["is_msvc"]
assert is_release == (env["target"] == "template_release")

if use_xsimd:
    env.Append(CCFLAGS=[
        # See https://xtensor.readthedocs.io/en/latest/build-options.html
        # See https://github.com/xtensor-stack/xsimd for supported list of simd extensions.
        # Choosing more will make your program faster, but also more incompatible to older machines.
        "-DXTENSOR_USE_XSIMD=1",
        # This adds some sanity checks which will throw if failed.
        # It claims to do bounds checks but it only does it VERY sparsely as of yet.
        "-DXTENSOR_ENABLE_ASSERT=1",
    ])

if env["platform"] == "windows":
    # At least the github runner needs bigobj to be enabled (otherwise it crashes).
    # is_msvc is set by godot-cpp.
    if is_msvc:
        env.Append(CCFLAGS=["/bigobj"])
    else:
        env.Append(CCFLAGS=["-Wa,-mbig-obj"])

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

# Filename of the library.
lib_filename = f"{env.subst('$SHLIBPREFIX')}{libname}.{env['platform']}.{env['arch']}{env.subst('$SHLIBSUFFIX')}"
# Build releases into build/, and debug into demo/.
lib_filepath = ""

if env["platform"] == "macos" or env["platform"] == "ios":
    # The above defaults to creating a .dylib.
    # These are not supported on the iOS app store.
    # To make it consistent, we'll just use frameworks on both macOS and iOS.
    framework_name = "{}.{}.{}".format(libname, env["platform"], env["arch"])
    lib_filename = framework_name
    lib_filepath = "{}.framework/".format(framework_name)

    env["SHLIBPREFIX"] = ""
    env["SHLIBSUFFIX"] = ""

library = env.SharedLibrary(
    os.path.join(env["build_dir"], f"addons/{libname}/{env['platform']}/{lib_filepath}{lib_filename}"),
    source=sources,
)

Default(library)
