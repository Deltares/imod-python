"""
Logging in iMOD Python
======================================

iMOD Python supports logging through both the standard
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

simulation = imod.data.hondsrug_simulation(tmpdir / "hondsrug_saved")
# %%
# If we run the command again with the log level set to DEBUG,
# we can see more detailed logging output:

imod.logging.configure(LoggerType.LOGURU, log_level=LogLevel.DEBUG)

simulation = imod.data.hondsrug_simulation(tmpdir / "hondsrug_saved")

# %%
# It is also possible to log the output to a default log file, `imod-python.log`,
# by adding `add_default_file_handler=True` to the command.
# Here we setup logging using the python logging framework. Notice how the output
# is slightly different than the Loguru output, but the information is similar.

imod.logging.configure(
    LoggerType.PYTHON, log_level=LogLevel.INFO, add_default_file_handler=True
)
simulation = imod.data.hondsrug_simulation(tmpdir / "hondsrug_saved")

# %%
# Sometimes, it might be useful to redirect logging to a specific file,
# such as when processing large datasets or running automated
# simulations, to keep logging output organised for debugging and verification.
# Here, we also set `add_default_stream_handler=True`, which controls if logging
# output is also sent to the console.

# Context manager to handle redirection of log output
from contextlib import redirect_stdout

logfile_path = "open_simulation.log"

with open(logfile_path, "w") as f:
    # Redirect stdout to the log file
    with redirect_stdout(f):
        # Configure logging
        imod.logging.configure(
            LoggerType.PYTHON,
            log_level=LogLevel.INFO,
            add_default_file_handler=False,
            add_default_stream_handler=True,
        )
        # Load the simulation
        simulation = imod.data.hondsrug_simulation(tmpdir / "hondsrug_saved")
