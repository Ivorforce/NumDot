#!/usr/bin/env python
import os
import sys

from methods import print_error


def normalize_path(val, env):
    return val if os.path.isabs(val) else os.path.join(env.Dir("#").abspath, val)


def validate_parent_dir(key, val, env):
    if not os.path.isdir(normalize_path(os.path.dirname(val), env)):
        raise UserError("'%s' is not a directory: %s" % (key, os.path.dirname(val)))


libname = "numdot"
projectdir = "demo"

localEnv = Environment(tools=["default"], PLATFORM="")

customs = ["custom.py"]
customs = [os.path.abspath(path) for path in customs]

opts = Variables(customs, ARGUMENTS)
opts.Add(
    BoolVariable(
        key="compiledb",
        help="Generate compilation DB (`compile_commands.json`) for external tools",
        default=localEnv.get("compiledb", False),
    )
)
opts.Add(
    PathVariable(
        key="compiledb_file",
        help="Path to a custom `compile_commands.json` file",
        default=localEnv.get("compiledb_file", "compile_commands.json"),
        validator=validate_parent_dir,
    )
)
opts.Update(localEnv)

Help(opts.GenerateHelpText(localEnv))

env = localEnv.Clone()
# To read up on why exceptions must be enabled, read further below.
env['disable_exceptions'] = False
env["compiledb"] = False

env.Tool("compilation_db")
compilation_db = env.CompilationDatabase(
    normalize_path(localEnv["compiledb_file"], localEnv)
)
env.Alias("compiledb", compilation_db)

submodule_initialized = False
dir_name = 'godot-cpp'
if os.path.isdir(dir_name):
    if os.listdir(dir_name):
        submodule_initialized = True

if not submodule_initialized:
    print_error("""godot-cpp is not available within this folder, as Git submodules haven't been initialized.
Run the following command to download godot-cpp:

    git submodule update --init --recursive""")
    sys.exit(1)

env = SConscript("godot-cpp/SConstruct", {"env": env, "customs": customs})

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

if env["target"] in ["editor", "template_debug"]:
    try:
        doc_data = env.GodotCPPDocData("src/gen/doc_data.gen.cpp", source=Glob("doc_classes/*.xml"))
        sources.append(doc_data)
    except AttributeError:
        print("Not including class reference as we're targeting a pre-4.3 baseline.")

file = "{}{}{}".format(libname, env["suffix"], env["SHLIBSUFFIX"])
filepath = ""

if env["platform"] == "macos" or env["platform"] == "ios":
    filepath = "{}.framework/".format(env["platform"])
    file = "{}.{}.{}".format(libname, env["platform"], env["target"])

libraryfile = "bin/{}/{}{}".format(env["platform"], filepath, file)
library = env.SharedLibrary(
    libraryfile,
    source=sources,
)

copy = env.InstallAs("{}/bin/{}/{}lib{}".format(projectdir, env["platform"], filepath, file), library)

default_args = [library, copy]
if localEnv.get("compiledb", False):
    default_args += [compilation_db]
Default(*default_args)
