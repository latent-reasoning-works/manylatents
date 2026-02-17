# Namespace package - this allows extensions like manylatents.omics to extend manylatents
__path__ = __import__('pkgutil').extend_path(__path__, __name__)

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0"
