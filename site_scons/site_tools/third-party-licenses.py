import pathlib


# Order roughly matches link/include relevance in the shipped binary.
DEPENDENCIES = [
	{
		"name": "godot-cpp",
		"url": "https://github.com/godotengine/godot-cpp",
		"license_path": "godot-cpp/LICENSE.md",
	},
	{
		"name": "xtensor",
		"url": "https://github.com/xtensor-stack/xtensor",
		"license_path": "xtensor/LICENSE",
	},
	{
		"name": "xtl",
		"url": "https://github.com/xtensor-stack/xtl",
		"license_path": "xtl/LICENSE",
	},
	{
		"name": "xsimd",
		"url": "https://github.com/xtensor-stack/xsimd",
		"license_path": "xsimd/LICENSE",
	},
	{
		"name": "xtensor-signal",
		"url": "https://github.com/xtensor-stack/xtensor-signal",
		"license_path": "xtensor-signal/LICENSE",
	},
]


def exists(env):
	return True


def options(opts):
	pass


def generate(target, *, env):
	target_path = pathlib.Path(target)
	source_paths = [pathlib.Path(d["license_path"]) for d in DEPENDENCIES]

	def write_licenses(target, source, env):
		out = [
			"# Third-Party Licenses",
			"",
			"NumDot's binary distribution incorporates source from the following "
			"third-party projects. Each project's full license text is reproduced below.",
			"",
		]
		for dep in DEPENDENCIES:
			license_text = pathlib.Path(dep["license_path"]).read_text().rstrip()
			out.append(f"## {dep['name']}")
			out.append("")
			out.append(dep["url"])
			out.append("")
			out.append("```")
			out.append(license_text)
			out.append("```")
			out.append("")
		pathlib.Path(target[0].path).write_text("\n".join(out))

	return env.Command(
		str(target_path), [str(p) for p in source_paths],
		action=write_licenses,
	)
