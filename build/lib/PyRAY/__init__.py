"""Core library imports for PyRAY."""

# Define a logger object to allow easier log handling
import logging
logging.raiseExceptions = False
logger = logging.getLogger('PyRAY_logger')

osflag = False
try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata
    osflag = True

# Import the package modules and top-level classes
from PyRAY import library  # noqa F401

# Set version
__version__ = metadata.version('PyRAY')
