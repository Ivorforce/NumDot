def exists(env):
    return True

def generate(env, features):
    # prints all features on the command line on compile.
    # print(features.all)

    # Disable all features
    features.disable(features.all)

    # Other feature groups you can enable or disable at once (using features.XXX):
    # bitwise, logical, trigonometry, random

    # Enable 'add' features
    features.enable('add')
