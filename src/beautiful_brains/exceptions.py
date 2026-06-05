"""Package-specific exceptions."""


class BeautifulBrainsError(Exception):
    """Base exception for user-facing Beautiful-Brains errors."""


class ConfigError(BeautifulBrainsError):
    """Raised when user-supplied configuration is invalid."""


class DependencyError(BeautifulBrainsError):
    """Raised when an optional runtime dependency is unavailable."""


class ImageGeometryError(BeautifulBrainsError):
    """Raised when images cannot be safely combined in the same voxel space."""
