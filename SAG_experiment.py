from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4
from parcels import Variable, ErrorCode, Field, DiffusionUniformKh
from datetime import timedelta
from datetime import datetime
from parcels import GeographicPolar, Geographic
import numpy as np
import xarray as xr
import sys
import local_kernels as kernels

series = 4
resusTime = 10
shoreTime = 10
n_points = 100  # particles per sampling site
n_days = 10  # 22*30  # number of days to simulate
K_bar = 10  # diffusion value
stored_dt = 1  # hours
loc = sys.argv[1]
# The file go from:
# 23 oct 2018 - 23 nov 2018
# 23 nov 2018 - 23 dic 2018
# 23 dic 2018 - 23 jan 2019

# data = '../data/mercatorpsy4v3r1_gl12_mean_20180101_R20180110.nc'
data = 'data/mercatorpsy4v3r1_gl12_mean_20180101_R20180110.nc'
output_path = f'data/sa-S{series:02d}.nc'

# loading the fields that have to do with the coastline.
coastal_fields = xr.load_dataset('../coastal_fields.nc')
# data = '/data/oceanparcels/input_data/CMEMS/' + \
#        'GLOBAL_ANALYSIS_FORECAST_PHY_001_024/*.nc'  # gemini
# output_path = f'/scratch/cpierard/source_{loc}_release.nc'

# time range 2018-01-01 to 2019-11-27
filesnames = {'U': data,
              'V': data}

variables = {'U': 'uo',
             'V': 'vo'}  # Use utotal

dimensions = {'lat': 'latitude',
              'lon': 'longitude',
              'time': 'time'}

indices = {'lat': range(*coastal_fields.index_lat),
           'lon': range(*coastal_fields.index_lon)}

fieldset = FieldSet.from_netcdf(filesnames, variables, dimensions,
                                allow_time_extrapolation=True, indices=indices)

###############################################################################
# Adding the border current, which applies for all scenarios except for 0     #
###############################################################################
u_border = coastal_fields.coastal_u.values
v_border = coastal_fields.coastal_u.values
fieldset.add_field(Field('borU', data=u_border,
                         lon=fieldset.U.grid.lon, lat=fieldset.U.grid.lat,
                         mesh='spherical'))
fieldset.add_field(Field('borV', data=v_border,
                         lon=fieldset.U.grid.lon, lat=fieldset.U.grid.lat,
                         mesh='spherical'))
fieldset.borU.units = GeographicPolar()
fieldset.borV.units = Geographic()

###############################################################################
# Adding in the  land cell identifiers                                        #
###############################################################################
landID = coastal_fields.landmask.values
fieldset.add_field(Field('landID', landID,
                         lon=fieldset.U.grid.lon, lat=fieldset.U.grid.lat,
                         mesh='spherical'))


###############################################################################
# Adding the horizontal diffusion                                             #
###############################################################################
size2D = (fieldset.U.grid.ydim, fieldset.U.grid.xdim)
K_bar = 10
K_h = K_bar * np.ones(size2D)
nx, ny = np.where(landID == 1)
K_h[nx, ny] = 0
fieldset.add_field(Field('Kh_zonal', data=K_h,
                         lon=fieldset.U.grid.lon, lat=fieldset.U.grid.lat,
                         mesh='spherical'))
fieldset.add_field(Field('Kh_meridional', data=K_h,
                         lon=fieldset.U.grid.lon, lat=fieldset.U.grid.lat,
                         mesh='spherical'))


###############################################################################
# Distance to the shore                                                       #
###############################################################################
distance = coastal_fields.distance2shore.values
fieldset.add_field(Field('distance2shore', distance,
                         lon=fieldset.U.grid.lon, lat=fieldset.U.grid.lat,
                         mesh='spherical'))


class SimpleBeachingResuspensionParticle(JITParticle):
    # Now the beaching variables
    # 0=open ocean, 1=beached
    beach = Variable('beach', dtype=np.int32,
                     initial=0)
    # Now the setting of the resuspension time and beaching time
    resus_t = Variable('resus_t', dtype=np.float32,
                       initial=resusTime, to_write=False)
    coastPar = Variable('coastPar', dtype=np.float32,
                        initial=shoreTime, to_write=False)
    # Finally, I want to keep track of the age of the particle
    age = Variable('age', dtype=np.int32, initial=0)
    # Weight of the particle in tons
    # weights = Variable('weights', dtype=np.float32,
    #                    initial=attrgetter('weights'))
    # Distance of the particle to the coast
    distance = Variable('distance', dtype=np.float32, initial=0)


#####################################
# Opening file with positions and sampling dates.
river_sources = np.load('river_sources.npy', allow_pickle=True).item()

np.random.seed(0)  # to repeat experiment in the same conditions
# Create the cluster of particles around the sampling site
# with a radius of 1/24 deg (?).
# time = datetime.datetime.strptime('2018-01-01 12:00:00', '%Y-%m-%d %H:%M:%S')

lon_cluster = [river_sources[loc][1]]*n_points
lat_cluster = [river_sources[loc][0]]*n_points
lon_cluster = np.array(lon_cluster)+(np.random.random(len(lon_cluster))-0.5)/24
lat_cluster = np.array(lat_cluster)+(np.random.random(len(lat_cluster))-0.5)/24
beached = np.zeros_like(lon_cluster)
age_par = np.zeros_like(lon_cluster)

start_time = datetime.strptime('2018-01-01 12:00:00',
                               '%Y-%m-%d %H:%M:%S')
# date_cluster = np.repeat(start_time, n_points)
date_cluster = np.empty(n_points, dtype='O')
for i in range(n_points):
    random_date = start_time + timedelta(hours=np.random.randint(0, 23))
    date_cluster[i] = random_date

# creating the Particle set
pset = ParticleSet.from_list(fieldset=fieldset,
                             pclass=SimpleBeachingResuspensionParticle,
                             lon=lon_cluster,
                             lat=lat_cluster,
                             time=date_cluster,
                             beach=beached, age=age_par)


###############################################################################
# And now the overall kernel                                                  #
###############################################################################
def delete_particle(particle, fieldset, time):
    particle.delete()


totalKernel = pset.Kernel(kernels.AdvectionRK4_floating) + \
    pset.Kernel(kernels.AntiBeachNudging) + \
    pset.Kernel(kernels.BrownianMotion2D) + \
    pset.Kernel(kernels.beach)

# Output file
output_file = pset.ParticleFile(
    name=output_path,
    outputdt=timedelta(hours=stored_dt))

# Execute!
pset.execute(totalKernel,
             runtime=timedelta(days=n_days),
             dt=timedelta(hours=1),
             output_file=output_file,
             recovery={ErrorCode.ErrorOutOfBounds: delete_particle})
output_file.close()
