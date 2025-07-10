from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("autoalign")
except PackageNotFoundError:
    __version__ = "unknown"