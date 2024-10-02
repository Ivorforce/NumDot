#include "register_types.hpp"

#include <gdextension_interface.h>      // for GDExtensionBool, GDExtensionC...
#include <godot_cpp/core/defs.hpp>      // for GDE_EXPORT
#include <godot_cpp/godot.hpp>          // for ModuleInitializationLevel
#include "godot_cpp/core/class_db.hpp"  // for GDREGISTER_CLASS
#include "nd.hpp"                         // for nd
#include "ndf.hpp"                         // for ndf
#include "ndb.hpp"                         // for ndb
#include "ndi.hpp"                         // for ndi
#include "ndarray.hpp"                    // for NDArray

using namespace godot;

void initialize_numdot_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	GDREGISTER_CLASS(nd);
	GDREGISTER_CLASS(ndf);
	GDREGISTER_CLASS(ndi);
	GDREGISTER_CLASS(ndb);
	GDREGISTER_CLASS(NDArray);
}

void uninitialize_numdot_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
}

extern "C" {
// Initialization.
GDExtensionBool GDE_EXPORT numdot_library_init(GDExtensionInterfaceGetProcAddress p_get_proc_address, const GDExtensionClassLibraryPtr p_library, GDExtensionInitialization* r_initialization) {
	godot::GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);

	init_obj.register_initializer(initialize_numdot_module);
	init_obj.register_terminator(uninitialize_numdot_module);
	init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);

	return init_obj.init();
}
}
