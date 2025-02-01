# Doc

## Install for developing documentation

```bash
conda create -n ata_doc --clone ata
pip install .[doc]
```

## Add new documentation

Add new markdown file in ```./docs/```.

## Preview the docs
``` bash
cd docs
rst2myst convert *.rst
sphinx-autobuild . _build/html
```