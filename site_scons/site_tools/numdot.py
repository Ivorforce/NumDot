import pathlib
from SCons.Tool import Tool
from SCons.Script import ARGLIST
from SCons.Variables import BoolVariable, PathVariable
from SCons.Variables.BoolVariable import _text2bool


def exists(env):
    return True

def options(opts):
    opts.Add(
        "use_xsimd",
        "Whether to use xsimd, accelerating contiguous memory computation. Defaults to no on web and yes elsewhere.",
        "auto"
    )
    opts.Add(
        "openmp_threshold",
        "If 0 or above, use OpenMP, for parallel assignment for operation sizes above or equal to the threshold. Defaults to -1 (no OpenMP).",
        "-1"
    )
    opts.Add(
        "optimize_for_arch",
        "Enable all optimizations the arch supports, making the build incompatible with other machines. Use 'native' to optimize for this machine. Note that on macOS, setting this option also requires setting arch= to a specific arch, e.g. arch=x86_64 or arch=arm64.",
        "",
    )

    features_tool = Tool("features")
    features2_tool = Tool("features2")
    scu_tool = Tool("scu")

    features_tool.options(opts)
    scu_tool.options(opts)

def generate(env, godot_cpp_env, sources):
    features_tool = Tool("features")
    features2_tool = Tool("features2")
    scu_tool = Tool("scu")

    # Allow additional defines, see https://scons.org/doc/production/HTML/scons-user/ch10s02.html
    cppdefines = []
    for key, value in ARGLIST:
        if key == 'define':
            cppdefines.append(value)

    optimize_for_arch = env["optimize_for_arch"]
    openmp_threshold = int(env["openmp_threshold"])

    if optimize_for_arch:
        # Yo-march improves performance, makes the build incompatible with most other machines.
        env.Append(CPPFLAGS=[f"-march={optimize_for_arch}"])

    is_msvc = "is_msvc" in env and env["is_msvc"]

    if env["use_xsimd"] == "auto":
        use_xsimd = True
    else:
        use_xsimd = _text2bool(env["use_xsimd"])

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

    if env['platform'] == "web" and use_xsimd:
        # Not enabled by default, and xsimd doesn't have guards against it so we have to force-add it.
        # See https://github.com/emscripten-core/emscripten/issues/12714.
        env.Append(CPPFLAGS=["-msimd128"])
        # TODO We could also pass -fno-vectorize for size-optimizing builds, as discussed in the linked issue.

    if openmp_threshold >= 0:
        # TODO Support is not yet complete. We somehow need include paths for each OS.
        if is_msvc:
            env.Append(CCFLAGS=['/openmp'])
            env.Append(LINKFLAGS=['/openmp'])
        else:
            env.Append(CCFLAGS=['-fopenmp'])
            env.Append(LINKFLAGS=['-fopenmp'])

        env.Append(CCFLAGS=["-DXTENSOR_USE_OPENMP", f"-DXTENSOR_OPENMP_TRESHOLD={openmp_threshold}"])

    env.Append(CPPPATH=["xtl/include", "xsimd/include", "xtensor/include", "xtensor-signal/include"])
    env.Append(CPPPATH=["src/"])

    sources.extend([
        f for f in env.Glob("src/*.cpp") + env.Glob("src/*/*.cpp") + env.Glob("src/*/*/*.cpp")
        # Generated files will be added selectively and maintained by tools.
        if not "/gen/" in str(f.path)
    ])
    # FIXME
    features2_tool.generate(env, sources)

    scu_tool.generate(env, sources)
    features_tool.generate(env)

    if env["target"] in ["editor", "template_debug"]:
        doc_data = env.GodotCPPDocData("src/gen/doc_data.gen.cpp", source=env.Glob("doc_classes/*.xml"))
        sources.append(doc_data)
