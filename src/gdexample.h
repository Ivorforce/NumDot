#ifndef GDEXAMPLE_H
#define GDEXAMPLE_H

#include <godot_cpp/classes/object.hpp>

namespace godot {

class GDExample : public Object {
	GDCLASS(GDExample, Object)

private:
	double time_passed;

protected:
	static void _bind_methods();

public:
	GDExample();
	~GDExample();
};

}

#endif
