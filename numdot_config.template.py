# This function has to exist for SCons reasons.
def exists(env):
    return True

def generate(env, features):
    """
    Modify the prevalent features in the current build.
    :param env: The SCons Environment object for this build.
    :param features: The Features object for this build. You can find its definition in ./site_scons/site_tools/features.py.
    """
    # Disable random functions
    # features.disable(features.random)

    # Enable just logical functions
    # features.disable(features.all)
    # features.enable(features.logical)

    # Enable only sum function
    # features.disable(features.all)
    # features.enable("sum")

    # Print all features.
    # print(features.all)
    # print(features.enabled)
    # print(features.disabled)

    pass
