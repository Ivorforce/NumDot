# Contributing to NumDot

New collaborators are very welcome! There's 3 quick ways to start:

- [Come by our Discord and have a chat with us](https://discord.gg/mwS2sW6V5M).
- [Browse the issues for those good for newcomers](https://github.com/Ivorforce/NumDot/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22). Open up a pull request once you're done.
- Continue reading the document for a quick introduction into the NumDot ecosystem.

## Understanding NumDot Technology

First off, NumDot is a [GDExtension](https://docs.godotengine.org/en/stable/tutorials/scripting/gdextension/what_is_gdextension.html), that is, an extension for the [Godot engine](https://godotengine.org). 
If you aren't familiar with them, I recommend doing the short [C++ GDExtension tutorial](https://docs.godotengine.org/en/stable/tutorials/scripting/gdextension/gdextension_cpp_example.html) to understand the tools involved.

NumDot itself is a very thin wrapper over [xtensor](https://xtensor.readthedocs.io/en/latest/index.html). That means little code is involved, but some of it involves fairly advanced concepts:

- **[Tensors and broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html):** If unfamiliar with these terms, we recommend [experimenting with NumPy first](https://numpy.org/doc/stable/user/quickstart.html). While NumPy is not involved, it is the most popular implementation of this concept.
- **[C++ Templates](https://www.google.com/search?client=safari&rls=en&q=C%2B%2B+templates&ie=UTF-8&oe=UTF-8):** To generate efficient code with few lines, NumDot makes use of C++ templates. They're essentially fancy generics, but are a bit harder to understand.
- **[C++ Variant](https://en.cppreference.com/w/cpp/utility/variant):** To offer support for different data types, NumDot uses `std::variant`.
- **[XTensor / XArray](https://xtensor.readthedocs.io/en/latest/getting_started.html):** XTensor is used for actually doing the math. That means you need to understand what it's doing to produce code that works with it.

You don't need to be proficient with all of these technologies to help! Check out the [open issues](https://github.com/Ivorforce/NumDot/issues) for anything that interests you. There is a lot left to do!

## Making a contribution

Setting up the project is covered on the docs: [Custom Builds](https://numdot.readthedocs.io/en/latest/setup/custom_build_setup.html)

Additionally, please make sure you adhere to code style guideline, and update documentation, as described below.

### Documentation

If you changed NumDot's public API, you should also update its documentation. Start by running the doctool:

```bash
# Update the .xml files from source
cd demo && godot --doctool ../ --gdextension-docs && cd ..
# Update the .rst files from .xml files
curl -sSL https://raw.githubusercontent.com/godotengine/godot/master/doc/tools/make_rst.py | python3 - -o "docs/classes" -l "en" doc_classes
```

Edit the new entries in `./doc_classes/` ([see the godot docs for more information](https://docs.godotengine.org/en/stable/tutorials/scripting/gdextension/gdextension_docs_system.html)).

### Include Policy

[include-what-you-use](https://github.com/include-what-you-use/include-what-you-use/) should be used to keep the includes tidy and explicit. This workflow is needed because code that works with one compiler can break down with another, due to differences in intrinsic headers. If you can't run it, don't worry about it. We can adjust includes after merging your changes.

```
# Run the tool, dump the results into iwyu.txt.
iwyu_tool.py -p . src > iwyu.txt
```

The resulting txt is _very_ explicit. Drop xsimd and intrinsics specific files you are offered, but most of the rest can be copied over as-is.

### Proposing your changes

When you're done, edit the changelog page under the 'Latest (Unstable)' heading in `/docs/setup/changelog`.

Make a commit:
```bash
# Check your changes first.
git diff
# If everything seems ok:
git add .
git commit -m "My change message."
git push
```

Finally, [make a Pull Request (PR)](https://github.com/Ivorforce/NumDot/compare). We will check your changes, make suggestions, and finally integrate your code into the project. Try to make sure you don't include any accidental changes, like editing the test file.

## Any Problems?

Please [come by our Discord and have a chat with us](https://discord.gg/mwS2sW6V5M). We are happy you want to help, and should be able to help you make a contribution.
