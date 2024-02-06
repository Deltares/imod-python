import xarray as xr
import numpy as np

# Model units
length_units = "meters"
time_units = "days"

# Model parameters
nlay = 1  # Number of layers
nrow = 31  # Number of rows
ncol = 46  # Number of columns
delr = 10.0  # Column width ($m$)
delc = 10.0  # Row width ($m$)
delz = 10.0  # Layer thickness ($m$)
shape = (nlay, nrow, ncol)
top = 10.0
dims = ("layer", "y", "x")

y = np.arange(delr*nrow, 0, -delr) 
x = np.arange(0, delc*ncol, delc)
coords = {"layer": [1], "y": y, "x": x, "dx": delc, "dy": -delr}
idomain = xr.DataArray(np.ones(shape, dtype=int), coords=coords, dims=dims)


top = 0.0  # Top of the model ($m$)
prsity = 0.3  # Porosity
perlen = 365  # Simulation time ($days$)
k11 = 1.0  # Horizontal hydraulic conductivity ($m/d$)
qwell = 1.0  # Volumetric injection rate ($m^3/d$)
cwell = 1000.0  # Concentration of injected water ($mg/L$)
al = 10.0  # Longitudinal dispersivity ($m$)
trpt = 0.3  # Ratio of transverse to longitudinal dispersivity

# Additional model input
perlen = [1, 365.0]
nper = len(perlen)
nstp = [2, 730]
tsmult = [1.0, 1.0]
sconc = 0.0
dt0 = 0.3
ath1 = al * trpt
dmcoef = 0.0

botm = [top - delz]  # Model geometry

k33 = k11  # Vertical hydraulic conductivity ($m/d$)
icelltype = 0

# Initial conditions
Lx = (ncol - 1) * delr
v = 1.0 / 3.0
prsity = 0.3
q = v * prsity
h1 = q * Lx
strt = np.zeros((nlay, nrow, ncol), dtype=float)
strt[0, :, 0] = h1

ibound_mf2k5 = np.ones((nlay, nrow, ncol), dtype=int)
ibound_mf2k5[0, :, 0] = -1
ibound_mf2k5[0, :, -1] = -1
idomain = np.ones((nlay, nrow, ncol), dtype=int)
icbund = 1
c0 = 0.0
cncspd = [[(0, 0, 0), c0]]
welspd = {0: [[0, 15, 15, qwell]]}  # Well pumping info for MF2K5
spd = {0: [0, 15, 15, cwell, 2]}  # Well pupming info for MT3DMS
#              (k,  i,  j),  flow, conc
spd_mf6 = {0: [[(0, 15, 15), qwell, cwell]]}  # MF6 pumping information

# Set solver parameter values (and related)
nouter, ninner = 100, 300
hclose, rclose, relax = 1e-6, 1e-6, 1.0
ttsmult = 1.0
percel = 1.0  # HMOC parameters in case they are invoked
itrack = 3  # HMOC
wd = 0.5  # HMOC
dceps = 1.0e-5  # HMOC
nplane = 1  # HMOC
npl = 0  # HMOC
nph = 16  # HMOC
npmin = 4  # HMOC
npmax = 32  # HMOC
dchmoc = 1.0e-3  # HMOC
nlsink = nplane  # HMOC
npsink = nph  # HMOC

# Time discretization
tdis_rc = []
tdis_rc.append((perlen, nstp, 1.0))