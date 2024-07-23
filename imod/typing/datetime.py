from datetime import datetime

import cftime
import numpy as np
import pandas as pd

api_datetimetype = datetime | np.datetime64 | str | pd.DatetimeIndex | cftime.datetime
internal_datetimetype = np.datetime64 | cftime.datetime
