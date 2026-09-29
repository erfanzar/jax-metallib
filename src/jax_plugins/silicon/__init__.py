# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JAX metallib plugin for the Apple MPS backend."""

import logging
import os
import sys
import warnings
from pathlib import Path

_logger = logging.getLogger(__name__)

# jaxlib version this plugin was built against (major.minor). Used for runtime
# compatibility checking.
_BUILT_FOR_JAXLIB = (0, 11)
_LIB_NAME = "libpjrt_plugin_silicon.dylib"


class SiliconPluginError(Exception):
    """Exception raised when Silicon plugin initialization fails."""

    pass


def _get_search_paths():
    """Return list of (path, description) tuples for library search."""
    pkg_dir = Path(__file__).parent
    project_root = pkg_dir.parent.parent.parent

    return [
        (pkg_dir / _LIB_NAME, "package directory (editable install)"),
        (pkg_dir / "lib" / _LIB_NAME, "package lib/ (wheel install)"),
        (
            project_root / "build" / "*" / "lib" / _LIB_NAME,
            "build/*/lib/ (cmake build)",
        ),
        (Path("/usr/local/lib") / _LIB_NAME, "/usr/local/lib/"),
        (Path("/opt/homebrew/lib") / _LIB_NAME, "/opt/homebrew/lib/"),
    ]


def _validate_path(path):
    """Check if a path exists and is a file.

    Args:
        path: Path to validate.

    Returns:
        True if path exists and is a file, False otherwise.
    """
    try:
        return Path(path).is_file()
    except (OSError, ValueError) as e:
        _logger.debug("Failed to validate path %s: %s", path, e)
        return False


def _find_library():
    """Find the pjrt_plugin_silicon shared library.

    Returns:
        Path to the library, or None if not found.
    """
    # Environment variable takes precedence. Keep the previous variable name as
    # a fallback for compatibility with older scripts.
    for env_var in ("JAX_METALLIB_LIBRARY_PATH", "JAX_SILICON_LIBRARY_PATH", "JAX_MPS_LIBRARY_PATH"):
        if env_var in os.environ:
            env_path = os.environ[env_var]
            if _validate_path(env_path):
                return env_path
            warnings.warn(
                f"{env_var} is set to '{env_path}', but the file does not exist or is not a file.",
                stacklevel=2,
            )
            break

    for path, _ in _get_search_paths():
        # Handle glob patterns
        if "*" in str(path):
            try:
                for match in Path("/").glob(str(path).lstrip("/")):
                    if _validate_path(match):
                        return str(match)
            except OSError as e:
                _logger.debug("Failed to check glob pattern %s: %s", path, e)
        elif _validate_path(path):
            return str(path)

    return None


def _check_jaxlib_version():
    """Check if the installed jaxlib version is compatible.

    Warns if the major.minor version doesn't match what the plugin was built for.
    """
    try:
        import jaxlib

        version_str = getattr(jaxlib, "__version__", None)
        if version_str is None:
            return

        parts = version_str.split(".")
        if len(parts) < 2:
            return

        installed = (int(parts[0]), int(parts[1]))
        if installed != _BUILT_FOR_JAXLIB:
            warnings.warn(
                f"jax-metallib was built for jaxlib {_BUILT_FOR_JAXLIB[0]}.{_BUILT_FOR_JAXLIB[1]}.x, "
                f"but jaxlib {version_str} is installed. This may cause compatibility "
                f"issues with StableHLO bytecode parsing. Consider installing jaxlib "
                f">={_BUILT_FOR_JAXLIB[0]}.{_BUILT_FOR_JAXLIB[1]}.x.",
                stacklevel=2,
            )
    except Exception:
        # Silently ignore version check failures - they are non-fatal.
        pass


def _mark_backend_nonexperimental(xb, backend_name):
    """Mark a backend as non-experimental in JAX's registry.

    This suppresses the "experimental backend" warning for plugin backends.
    """
    try:
        factory = xb._backends[backend_name]
        if hasattr(factory, "options"):
            factory.options["available"] = True
    except (KeyError, TypeError):
        # Backend not yet registered or doesn't support options - ignore.
        pass


def initialize():
    """Initialize the Silicon plugin with JAX.

    This function is called by JAX's plugin discovery mechanism.

    Raises:
        SiliconPluginError: If Metal GPU is not available or plugin initialization fails.
    """
    # Check platform first
    if sys.platform != "darwin":
        raise SiliconPluginError(
            f"MPS plugin requires macOS, but running on {sys.platform}. MPS "
            "Performance Shaders) is only available on Apple devices."
        )

    # Check jaxlib version compatibility
    _check_jaxlib_version()

    library_path = _find_library()
    if library_path is None:
        searched = "\n".join(f"  - {desc}" for _, desc in _get_search_paths())
        raise SiliconPluginError(
            f"Could not find {_LIB_NAME}. Searched paths:\n{searched}\nYou can also set JAX_METALLIB_LIBRARY_PATH."
        )

    # Disable shardy partitioner - it produces sdy dialect ops that our StableHLO parser
    # doesn't support yet (JAX 0.10+ enables it by default)
    try:
        import jax

        jax.config.update("jax_use_shardy_partitioner", False)
    except Exception as e:
        warnings.warn(
            f"Failed to disable shardy partitioner: {e}. Some operations may not work correctly.",
            stacklevel=2,
        )

    # Register the plugin using JAX's xla_bridge API
    try:
        from jax._src import xla_bridge as xb
    except ImportError as e:
        raise SiliconPluginError(f"Failed to import JAX xla_bridge: {e}") from e

    if not hasattr(xb, "register_plugin"):
        raise SiliconPluginError("JAX version does not support register_plugin API.")

    try:
        # JAX currently emits an "experimental backend" warning for plugin backends
        # unless they are listed as non-experimental.
        _mark_backend_nonexperimental(xb, "mps")
        xb.register_plugin(
            "mps",
            priority=500,
            library_path=library_path,
            options={},
        )
    except Exception as e:
        raise SiliconPluginError(f"Failed to register MPS plugin: {e}") from e

    # JAX has no eigh lowering rule for platform "mps" (only CPU/GPU/TPU), so
    # jnp.linalg.eigh/eigvalsh failed at lowering time. Register pure-JAX
    # lowerings now that "mps" is a known platform.
    try:
        from . import linalg_lowerings

        linalg_lowerings.register()
    except Exception as e:
        warnings.warn(f"Failed to register MPS linalg lowerings: {e}", stacklevel=2)
