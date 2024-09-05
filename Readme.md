# NumDot

A tensor math library for the [Godot](https://godotengine.org) engine. Proof of concept, [contributions are welcome](https://github.com/Ivorforce/NumDot/blob/main/CONTRIBUTING.md).

NumDot uses [xtensor](https://github.com/xtensor-stack/xtensor) under the hood.

### What and Why?

A tensor (or nd-array) is essentially a multi-dimensional array. You can run mathematical operations on the _whole tensor_ at once, making these operations very fast. Currently, common math operations can run up to 30 times faster than a conventional for-loop in gdscript (when using about 100+ floats).

Operations with tensors are also very easy to write and read, once you're familiar with them. Vectorization allows you to forget about for loops, and focus on the operation itself.

NumDot is inspired by the python tensor math library, [NumPy](https://numpy.org), and thus shares many semantics with it. If you are unfamiliar with tensor operations in general, I recommend taking a numpy tutorial or two first. That being said, here are some direct comparisons:

| NumPy  | NumDot |
| ------------- | ------------- |
| `x[a, b, c]` | `x.get(a, b, c)` |
| `x[...]` can return views | `x.get(...)` ([regretfully](https://github.com/Ivorforce/NumDot/issues/8)) returns a copy |
| `x[...]` can return single values | `x.get(...)` always returns a tensor, use `x.get_float(...)` and `x.get_int(...)` to get values. |
| `x[a, b, c] = d` | `x.set(d, a, b, c)` |
| `x[1:]` | `x.get(nd.from(1))` |
| `x[:1]` | `x.get(nd.to(1))` |
| `x[1:2]` | `x.get(nd.range(1, 2))` |
| `x[0:5:2]` | `x.get(nd.range_step(0, 5, 2))` |
| `np.array([2, 3, 4])` | `nd.array([2, 3, 4])` |
| `np.ones((2, 3, 4))` | `nd.ones([2, 3, 4])` |
| `np.add(a, b)` | `nd.add(a, b)` |
| `np.sin(a)` | `nd.sin(a)` |

Keep in mind these semantics are yet subject to change.

#### Godot Interoperability

Godot types are automatically converted to NumDot types for operations. You can also convert it back to godot types:
```gdscript
var a = nd.array(PackedFloat32Array([1, 2, 3]))
a = nd.add(a, 5)
var b: PackedFloat32Array = a.to_packed_float32_array()
```


### Installing and / or building

To add this extension into your project, [run the following](https://docs.godotengine.org/en/stable/tutorials/scripting/gdextension/gdextension_cpp_example.html):
```bash
cd your/project/folder
git submodule add https://github.com/Ivorforce/NumDot
cd numdot
git submodule update --init
cd ..
```

Build the library like so:

```bash
# Need to do this once at the start
cd godot-cpp
# Replace Use the fitting platform name of [macos, windows, linux]
scons platform=<platform> custom_api_file=../extension_api.json
cd ..

# Exceptions have to be explicitly enabled here
scons platform=macos
# You may have to build twice, see https://github.com/Ivorforce/NumDot/issues/23
```

### What Now?

NumDot is still in its experimental stages. If you want to keep up to date, come by and chat with us on our [Discord Server](https://discord.gg/hxuWcAXF).

If you want to contribute, check out [CONTRIBUTING.md](https://github.com/Ivorforce/NumDot/blob/main/CONTRIBUTING.md) or come by our [Discord Server](https://discord.gg/hxuWcAXF).

We will be keeping track of ToDos through the GitHub issues.
