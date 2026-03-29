# Namespace package - this allows extensions like manylatents.omics to extend manylatents
__path__ = __import__('pkgutil').extend_path(__path__, __name__)

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0"

# Lazy imports — avoids eagerly loading metric registry on every `import manylatents`.
# Only intercept known names; unknown names must raise AttributeError so Python's
# import machinery falls back to pkgutil.extend_path for namespace subpackages
# (e.g., manylatents.omics).
_LAZY_IMPORTS = {
    "evaluate_metrics": "manylatents.evaluate",
    "evaluate": "manylatents.evaluate",
}

def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
