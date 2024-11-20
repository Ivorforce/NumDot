#!/usr/bin/env python
import os
import sys
from pathlib import Path
from SCons.Variables.BoolVariable import _text2bool

from methods import print_error, print_warning

if not (os.path.isdir("godot-cpp") and os.listdir("godot-cpp")):
    print_error("""godot-cpp is not available within this folder, as Git submodules haven"t been initialized.
Run the following command to download godot-cpp:

    git submodule update --init --recursive""")
    sys.exit(1)

# ============================= Project Setup =============================

libname = "numdot"

# Load variables from custom.py, in case someone wants to store their own arguments.
# See https://scons.org/doc/production/HTML/scons-user.html#app-tools // search custom.py
customs = ["custom.py"]
customs = [os.path.abspath(path) for path in customs]
opts = Variables(customs, ARGUMENTS)

opts.Add(
    PathVariable(
        key="install_dir",
        help="Optional target project location the binary. The binary always builds in build/, but if this argument is supplied, the binary will be copied to the target location.",
        default=None,
    )
)

numdot_tool = Tool("numdot")
numdot_tool.options(opts)

# Only used to evaluate our own options, lol
options_env = Environment(tools=["default"], PLATFORM="")
opts.Update(options_env)
Help(opts.GenerateHelpText(options_env))

# Remove our custom options to avoid passing to godot-cpp; godot-cpp has its own check for unknown options.
for opt in opts.options:
    ARGUMENTS.pop(opt.key, None)

# ============================= Change defaults of godot-cpp =============================

if ARGUMENTS.get("platform", None) == "web":
    ARGUMENTS.setdefault("threads", "no")
    if _text2bool(ARGUMENTS.get("threads", "yes")):
        # TODO Figure out why that is. Does godot default to no threads exports?
        raise ValueError("NumDot does not currently support compiling web with threads.")

# To read up on why exceptions should be enabled, read further below.
if ARGUMENTS.get("disable_exceptions", None):
    raise ValueError("NumDot does not currently support compiling without exceptions.")
ARGUMENTS["disable_exceptions"] = False

# Clarification: template_debug and template_release are, from our perspective, both releases.
# template_debug is just for in-editor
is_release = not _text2bool(ARGUMENTS.get("dev_build", "no"))

if ARGUMENTS.get("optimize", None) is None and is_release:
    # The default godot-cpp optimizes for speed for release builds.
    if ARGUMENTS.get("platform", None) == "web" and ARGUMENTS.get("target", "template_debug") == "template_release":
        # For web, optimize binary size, can shrink by ~30%.
        ARGUMENTS["optimize"] = "size"

# Load godot-cpp
godot_cpp_env = SConscript("godot-cpp/SConstruct", {"customs": customs})

env = godot_cpp_env.Clone()
for opt in opts.options:
    if opt.key in options_env:
        env[opt.key] = options_env[opt.key]

is_msvc = "is_msvc" in env and env["is_msvc"]

# ============================= Actual source and lib setup =============================

# TODO Can replace when https://github.com/godotengine/godot-cpp/pull/1601 is merged.
if is_release:
    # Enable link-time optimization.
    # This further lets the compiler optimize, reduce binary size (~.5mb) or inline functions (possibly improving speeds).
    if is_msvc:
        env.Append(CCFLAGS=["/GL"])
        env.Append(LINKFLAGS=["/LTCG"])
    else:
        env.Append(CCFLAGS=["-flto"])
        env.Append(LINKFLAGS=["-flto"])

sources = []
targets = []

numdot_tool.generate(env, godot_cpp_env, sources)

# .dev doesn't inhibit compatibility, so we don't need to key it.
# .universal just means "compatible with all relevant arches" so we don't need to key it.
suffix = env['suffix'].replace(".dev", "").replace(".universal", "")

# Filename of the library.
lib_filename = f"{env.subst('$SHLIBPREFIX')}{libname}{suffix}{env.subst('$SHLIBSUFFIX')}"
# Build releases into build/, and debug into demo/.
lib_filepath = ""

if env["platform"] == "macos" or env["platform"] == "ios":
    # The above defaults to creating a .dylib.
    # These are not supported on the iOS app store.
    # To make it consistent, we'll just use frameworks on both macOS and iOS.
    framework_name = f"{libname}{suffix}"
    lib_filename = framework_name
    lib_filepath = "{}.framework/".format(framework_name)

    env["SHLIBPREFIX"] = ""
    env["SHLIBSUFFIX"] = ""

library = env.SharedLibrary(
    f"build/addons/{libname}/{env['platform']}/{lib_filepath}{lib_filename}",
    source=sources,
)
targets.append(library)

if env.get("install_dir", None) is not None:
    targets.append(env.Install(f"{env['install_dir']}/addons/{libname}/{env['platform']}/{lib_filepath}", library))

Default(targets)
