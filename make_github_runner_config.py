import json

# Ah, imagine if github matrix interpreters could just have this little function...
def unroll_config(cfg: dict):
    configs = [{}]

    for key, value in cfg.items():
        if key == "combine":
            # List of configs
            configs = [
                {**cfg_base, **cfg_combine}
                for cfg_base in configs
                for combine_template in cfg["combine"]
                for cfg_combine in unroll_config(combine_template)
            ]
        elif isinstance(value, list):
            # Value list
            configs = [
                {**config, key: value}
                for value in value
                for config in configs
            ]
        else:
            # Single value
            configs = [
                {**config, key: value}
                for config in configs
            ]

    return configs


config = dict(
    # Could build with double, but that would double (hah) our build count.
    # Doubles are custom builds, so let somebody else do that.
    float_precision=["single"],
    build_target_type=["template_debug", "template_release"],
    combine=[
        dict(
            platform="macos",
            os="macos-latest",
            arch="universal",
        ),
        dict(
            platform="ios",
            os="macos-latest",
            arch="arm64",
        ),
        dict(
            platform="linux",
            os="ubuntu-20.04",
            arch=["x86_64", "arm64", "rv64"],
        ),
        dict(
            platform="windows",
            os="windows-latest",
            arch=["x86_32", "x86_64"],
        ),
        dict(
            platform="android",
            os="ubuntu-20.04",
            arch=["x86_64", "arm64"],
        ),
        dict(
            platform="web",
            os="ubuntu-20.04",
            arch="wasm32",
        ),
    ]
)

includes = unroll_config(config)

print(json.dumps(includes))
