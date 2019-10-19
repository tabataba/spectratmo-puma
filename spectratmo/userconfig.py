"""User configuration (:mod:`spectratmo.userconfig`)
====================================================

Execute some user configuration files if they exist and gather the
configuration values as module attributes.

.. autoclass:: ConfigError
   :members:
   :private-members:

"""

from fluiddyn.util.userconfig import load_user_conf_files

config = load_user_conf_files('spectratmo')
del load_user_conf_files

glob = globals()
for k, v in config.items():
    glob[k] = v
del glob, k, v


class ConfigError(Exception):
    """A configuration error."""
