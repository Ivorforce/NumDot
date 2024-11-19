import os
import pathlib
import platform
import re
import sys
from typing import Collection

from SCons.Action import Action
from SCons.Builder import Builder
from SCons.Errors import UserError
from SCons.Script import ARGUMENTS
from SCons.Tool import Tool
from SCons.Variables import BoolVariable, EnumVariable, PathVariable
from SCons.Variables.BoolVariable import _text2bool

class Features:
    def __init__(self):
        self.mappings = dict()

    def set_is_enabled(self, features, is_enabled: bool):
        is_enabled = bool(is_enabled)

        if isinstance(features, str):
            self.mappings[features] = is_enabled
        elif isinstance(features, Collection):
            self.mappings.update({key: is_enabled for key in features})
        else:
            raise ValueError("Features must be string or a collection of features.")

    def enable(self, features):
        self.set_is_enabled(features, True)

    def disable(self, features):
        self.set_is_enabled(features, False)

def exists(env):
    return True

def options(opts):
    opts.Add(
        PathVariable(
            key="numdot_config",
            help="Path to a .py file that sets up custom NumDot configuration.",
            default=None,
        )
    )

def bool_to_string(val):
    return "true" if val else "false"

def generate(env):
    if 'numdot_config' not in env or not env['numdot_config']:
        return

    features = Features()

    custom_config_path = pathlib.Path(env['numdot_config'])
    assert custom_config_path.suffix == ".py"
    custom_config_tool = Tool(custom_config_path.with_suffix("").name, toolpath=[custom_config_path.parent])

    features_hpp_text = pathlib.Path("src/vatensor/vfeature.hpp").read_text()
    features_expr = re.search(r"enum\s+class\s+Feature\s*{([^}]*)}", features_hpp_text, re.MULTILINE)
    features.all = [
        f_
        for f in features_expr.group(1).split(",")
        for f_ in [f.strip()]
        if f_ and f_ != "count"
    ]

    features.bitwise = [f for f in features.all if f.startswith("bitwise")]
    features.logical = [f for f in features.all if f.startswith("logical")]
    features.trigonometry = [f for f in features.all if re.match(r"a?(sin|cos|tan)h?", f)]
    features.random = [f for f in features.all if f.startswith("random")]

    custom_config_tool.generate(env, features)

    dest_path = pathlib.Path("src/vatensor/gen/userconfig.hpp")
    dest_path.parent.mkdir(exist_ok=True)

    features_lines = ",\n".join(f"\t\t{{ Feature::{feature}, {bool_to_string(value)} }}" for feature, value in features.mappings.items())

    dest_path.write_text(
f"""#ifndef USERCONFIG_HPP
#define USERCONFIG_HPP

#include "../vfeature.hpp"

namespace va::userconfig {{
	constexpr std::initializer_list<std::pair<Feature, bool>> is_enabled_by_feature_initializer = {{
{features_lines}
	}};
}}

#endif //USERCONFIG_HPP
"""
    )

    env.Append(CPPDEFINES=["NUMDOT_USE_USER_CONFIG"])
