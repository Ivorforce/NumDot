import pathlib


def exists(env):
	return True


def options(opts):
	pass


def generate(target, *, env, name: str, description: str, author: str, version: str):
	target_path = pathlib.Path(target)

	if target_path.name != "plugin.cfg":
		raise ValueError("target must be some 'plugin.cfg'.")

	# See editor_plugin_settings.cpp
	def create_plugin_cfg(target, source, env):
		pathlib.Path(target[0].path).write_text(f"""\
[plugin]

name="{name}"
description="{description}"
author="{author}"
version="{version}"
script="plugin.gd"
language="C++"
""")

	# TODO For now, this appears to be required, unfortunately
	# TODO Make it disableable.
	def create_dummy_script(target, source, env):
		pathlib.Path(target[0].path).write_text(f"""\
@tool
extends EditorPlugin
""")

	return [
		env.Command(
			target, [],
			action=create_plugin_cfg
		),
		env.Command(
			str(target_path.parent / "plugin.gd"), [],
			action=create_dummy_script
		)
	]
