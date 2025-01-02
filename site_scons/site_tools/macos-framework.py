import pathlib


def exists(env):
	return True


def options(opts):
	pass


def generate(target, *, env, source, bundle_identifier: str, min_macos_version="10.12", min_ios_version="12.0"):
	if env["platform"] == "macos":
		dt_platform_name = "macosx"
		min_os_part = f"""
	<key>LSMinimumSystemVersion</key>
	<string>{min_macos_version}</string>\
"""
		plist_subpath = pathlib.Path("Resources/Info.plist")
	elif env["platform"] == "ios":
		dt_platform_name = "iphoneos"
		min_os_part = f"""
	<key>MinimumOSVersion</key>
	<string>{min_ios_version}</string>\
"""
		plist_subpath = pathlib.Path("Info.plist")
	else:
		return

	framework_path = pathlib.Path(target)
	framework_name = framework_path.name.removesuffix(".framework")

	# This is required because they affect the binary name, which we need to be equal to the framework name.
	env["SHLIBPREFIX"] = ""
	env["SHLIBSUFFIX"] = ""

	def create_info_plist(target, source, env):
		pathlib.Path(target[0].path).write_text(f"""\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleInfoDictionaryVersion</key>
	<string>6.0</string>
	<key>CFBundleDevelopmentRegion</key>
	<string>en</string>
	<key>CFBundleExecutable</key>
	<string>{framework_name}</string>
	<key>CFBundleName</key>
	<string>NumDot</string>
	<key>CFBundleDisplayName</key>
	<string>NumDot</string>
	<key>CFBundleIdentifier</key>
	<string>{bundle_identifier}</string>
	<key>NSHumanReadableCopyright</key>
	<string>Unlicensed</string>
	<key>CFBundleVersion</key>
	<string>1.0.0</string>
	<key>CFBundleShortVersionString</key>
	<string>1.0.0</string>
	<key>CFBundlePackageType</key>
	<string>FMWK</string>
	<key>CSResourcesFileMapped</key>
	<true/>
	<key>DTPlatformName</key>
	<string>{dt_platform_name}</string>{min_os_part}
</dict>
</plist>
""")

	return [
		env.SharedLibrary(
			str(framework_path / framework_name),
			source=source
		),
		env.Command(
			str(framework_path / plist_subpath), [],
			action=create_info_plist
		)
	]
