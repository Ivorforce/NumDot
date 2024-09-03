# NumDot

A tensor math library for the [Godot](https://godotengine.org) engine. Proof of concept. I will likely return to this soon and create a full release.

NumDot uses [xtensor](https://github.com/xtensor-stack/xtensor) under the hood.

### Building

Build the library like so:

```bash
# Replace Use the fitting platform name of [macos, windows, linux]
# Exceptions have to be explicitly enabled here
scons platform=macos
```

### Using

A tensor is essentially a multi-dimensional vector. You can run mathematical operations on the _whole tensor_ at once, making these operations very fast.

NumDot is inspired by the python tensor math library, [NumPy](https://numpy.org), and thus shares many semantics with it. If you are unfamiliar with tensor operations in general, I recommend taking a numpy tutorial or two first. That being said, here are some direct comparisons:

| NumPy  | NumDot |
| ------------- | ------------- |
| `x[a, b, c]` | `x.get(a, b, c)` |
| `x[a]` returns a view | `x[a]` returns a copy |
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

### Godot Interoperability

Godot types are automatically converted to NumDot types for operations. You can also convert it back to godot types:
```gdscript
var a = nd.array(PackedFloat32Array([1, 2, 3]))
a = nd.add(a, 5)
var b: PackedFloat32Array = a.to_packed_float32_array()
```

### What Now?

NumDot is still in its experimental stages. If you want to contribute, or enquire about the project's state, feel free to join on Discord: [Discord Channel](https://discord.gg/hxuWcAXF).

I will be keeping track of ToDos through the GitHub issues.
