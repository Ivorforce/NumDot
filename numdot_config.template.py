def exists(env):
    return True

def generate(env, features):
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
