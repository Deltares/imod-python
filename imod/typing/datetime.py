from datetime import datetime
import numpy as np
import pandas as pd
import cftime

api_datetimetype = datetime | np.datetime64 | str | pd.DatetimeIndex | cftime.datetime
internal_datetimetype = np.datetime64 | cftime.datetime