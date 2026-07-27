"""
Logging in iMOD-Python
======================================

iMod-python supports logging through both the standard
Python logging framework and Loguru, so that you can choose
whichever best fits your needs and project. By default,
logging is silent, so messages are only output once a logger
is configured.

In this example, we will use loading in the hondsrug simulation
to demonstrate logging capabilities.

"""

import imod
from imod.logging import LoggerType, LogLevel

# Create a temporary directory
tmpdir = imod.util.temporary_directory()

# %%
#
# Fetching an iMOD5 model
# -----------------------
#
# You can set up the logger by calling
# imod.logging.configure and choosing the
# type of logger (PYTHON, LOGURU) you would
# like to use:

imod.logging.configure(LoggerType.LOGURU)

# %%
# Additionally, you can set the level of logging you want
# (DEBUG, INFO, WARNING, ERROR, CRITICAL),
# where the default is WARNING. Here, we will use the Loguru
# logger and set the level to INFO so that we can see some
# logging output:

imod.logging.configure(LoggerType.LOGURU, log_level=LogLevel.INFO)

original_simulation = imod.data.hondsrug_simulation(tmpdir / "hondsrug_saved")
# %%

# %%
# We can then run this again but this time using the Python
# logging framework, as well as logging output to a file
# called `imod-python.log`, by adding
# `add_default_file_handler=True` to the command.

# Setup imod python logging using the python logging framework and write the log output to a file

imod.logging.configure(
    LoggerType.PYTHON, log_level=LogLevel.INFO, add_default_file_handler=True
)
original_simulation = imod.data.hondsrug_simulation(tmpdir / "hondsrug_saved")
