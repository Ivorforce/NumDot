# NumDot

A tensor math library for the [Godot](https://godotengine.org) engine. Work in Progress.

### Building

Build the library like so:

```bash
# Replace Use the fitting platform name of [macos, windows, linux]
# Exceptions have to be explicitly enabled here
scons platform=macos
```

### Notes

- As the size grows quadratically with combinations of operator x a_type x b_type, it may be beneficial to switch to promote-then-call.
    - This may worsen the runtime performance somewhat (up to testing), but it should speed up compilation times and binary size substantially.
