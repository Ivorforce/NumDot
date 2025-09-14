#!/usr/bin/env python
import os
import pathlib
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
        help="Target location for the addon. It will be appended with addons/numdot/ automatically.",
        default="build",
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

if ARGUMENTS.get("lto", None) is None and is_release:
    # Link-time optimization further lets the compiler optimize, reduce binary size (~.5mb) or inline functions (possibly improving speeds).
    # lto=auto disables LTO by default in some configurations. For us, it brings substantial improvements for configurations.
    ARGUMENTS["lto"] = "full"

if ARGUMENTS.get("build_profile", None) is None:
    # This improves compile time quite a lot, and can reduce the binary size too.
    ARGUMENTS["build_profile"] = str(pathlib.Path().parent / "configure" / "build_profile.json")

if ARGUMENTS.get("ndk_version", None) is None:
    # Need a newer ndk version for C++20 (r26+)
    ARGUMENTS["ndk_version"] = "27.2.12479018"

# Load godot-cpp
godot_cpp_env = SConscript("godot-cpp/SConstruct", {"customs": customs})

env = godot_cpp_env.Clone()
for opt in opts.options:
    if opt.key in options_env:
        env[opt.key] = options_env[opt.key]

is_msvc = "is_msvc" in env and env["is_msvc"]

# ============================= Actual source and lib setup =============================

sources = []
targets = []

numdot_tool.generate(env, godot_cpp_env, sources)

# .dev doesn't inhibit compatibility, so we don't need to key it.
# .universal just means "compatible with all relevant arches" so we don't need to key it.
suffix = env['suffix'].replace(".dev", "").replace(".universal", "")

addon_dir = f"{env['install_dir']}/addons/{libname}"
platform_dir = f"{addon_dir}/{env['platform']}"

lib_filename = f"{env.subst('$SHLIBPREFIX')}{libname}{suffix}{env.subst('$SHLIBSUFFIX')}"
library_targets = env.SharedLibrary(
    f"{platform_dir}/{lib_filename}",
    source=sources,
)

plugin_tool = Tool("plugin-cfg")
targets.append(plugin_tool.generate(
    f"{addon_dir}/plugin.cfg",
    env=env,
    name="NumDot",
    description="NumDot is a tensor math and scientific computation library for the Godot Engine.",
    author="Lukas Tenbrink and NumDot contributors.",
    version="0.11",
))

targets.extend(library_targets)
# Don't remove the file while building.
env.Precious(library_targets)

Default(targets)
