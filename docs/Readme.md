# NumDot Docs Readme

## Setup

You can install the project with pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r godot-docs/requirements.txt
```

## Build

To build the documentation, run:

```bash
source .venv/bin/activate
# Build class ref
cd .. && curl -sSL https://raw.githubusercontent.com/godotengine/godot/master/doc/tools/make_rst.py | python3 - -o "doc/classes" -l "en" doc_classes
# Build sphinx
cd doc && make html
```
