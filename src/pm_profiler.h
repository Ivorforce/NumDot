#ifndef NUMDOT_PM_PROFILER_H
#define NUMDOT_PM_PROFILER_H

#include <godot_cpp/variant/utility_functions.hpp>

#define GD_PROFILE_START std::chrono::high_resolution_clock::now();
#define GD_PRINT_DURATION(start) godot::UtilityFunctions::print(uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - (start)).count()));

#endif
