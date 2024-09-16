# NumDot Docs Readme

## Setup

You can install the project with pip:

```bash
pip install .
# Or, if poetry is preferred:
poetry install
```

## Build

To build the documentation, run:

```bash
# If using poetry:
poetry shell
# Build class ref
cd .. && curl -sSL https://raw.githubusercontent.com/godotengine/godot/master/doc/tools/make_rst.py | python3 - -o "doc/classes" -l "en" doc_classes
# Build sphinx
cd doc && make html
```
