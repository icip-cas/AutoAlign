import importlib
import os

PROJ_BASE_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def get_package_path(package_name):
    try:
        package = importlib.import_module(package_name)

        if hasattr(package, "__file__"):
            return os.path.dirname(os.path.abspath(package.__file__))
        elif hasattr(package, "__path__"):
            return package.__path__[0]
        else:
            return f"Cannot determine path for package '{package_name}'"
    except ImportError:
        return f"Package '{package_name}' not found or cannot be imported"
    except Exception as e:
        return f"Error finding package '{package_name}': {str(e)}"
