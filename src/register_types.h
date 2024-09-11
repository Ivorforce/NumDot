#ifndef NUMDOT_REGISTER_TYPES_H
#define NUMDOT_REGISTER_TYPES_H

#include "godot_cpp/godot.hpp"  // for ModuleInitializationLevel

using namespace godot;

void initialize_example_module(ModuleInitializationLevel p_level);
void uninitialize_example_module(ModuleInitializationLevel p_level);

#endif // NUMDOT_REGISTER_TYPES_H
