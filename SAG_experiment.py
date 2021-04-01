from parcels import FieldSet, ParticleSet, JITParticle
from parcels import Variable, ErrorCode, Field
from datetime import timedelta
from datetime import datetime
import numpy as np
import sys
from parcels import ParcelsRandom
import math
from parcels import GeographicPolar, Geographic
import local_kernels as kernels

resusTime = 69
shoreTime = 10
start_time = datetime.strptime('2016-04-01 12:00:00',
                               '%Y-%m-%d %H:%M:%S')
# end_time = '2020-08-31'
# delta = 1613 days
n_points = 100000  # particles per sampling site
n_days = 1600  # number of days to simulate
K_bar = 10  # diffusion value
stored_dt = 24  # hours
loc = sys.argv[1]

# data = '../data/mercatorpsy4v3r1_gl12_mean_20180101_R20180110.nc'
data = '/data/oceanparcels/input_data/CMEMS/' + \
    'GLOBAL_ANALYSIS_FORECAST_PHY_001_024_SMOC/*.nc'  # Gemini hourly
output_path = f'/scratch/cpierard/br-cr_{loc}_D{n_days}_N{n_points}.nc'

# time range 2018-01-01 to 2019-11-27
filesnames = {'U': data,
              'V': data}

variables = {'U': 'utotal',
             'V': 'vtotal'}  # Use utotal

dimensions = {'lat': 'latitude',
              'lon': 'longitude',
              'time': 'time'}
indices = {'lat': range(1, 900), 'lon': range(1284, 2460)}
fieldset = FieldSet.from_netcdf(filesnames, variables, dimensions,
                                allow_time_extrapolation=True, indices=indices)

###############################################################################
# Adding the border current, which applies for all scenarios except for 0     #
###############################################################################
u_border = np.load('coastal_u.npy')
v_border = np.load('coastal_v.npy')
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
landID = np.load('landmask.npy')
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
distance = np.load('distance2shore.npy')
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

date_cluster = np.empty(n_points, dtype='O')
for i in range(n_points):
    random_date = start_time + timedelta(days=np.random.randint(0, 365),
                                         hours=np.random.randint(0, 23))
    date_cluster[i] = random_date

# creating the Particle set
pset = ParticleSet.from_list(fieldset=fieldset,
                             pclass=SimpleBeachingResuspensionParticle,
                             lon=lon_cluster,
                             lat=lat_cluster,
                             time=date_cluster,
                             beach=beached, age=age_par)
###############################################################################
# Kernels                                                                     #
###############################################################################


def delete_particle(particle, fieldset, time):
    particle.delete()


def AdvectionRK4_floating(particle, fieldset, time):
    """Advection of particles using fourth-order Runge-Kutta integration.
    Function needs to be converted to Kernel object before execution
    A particle only moves if it has not beached (rather obviously)
    """
    if particle.beach == 0:
        particle.distance = fieldset.distance2shore[time, particle.depth,
                                                    particle.lat, particle.lon]
        (u1, v1) = fieldset.UV[time, particle.depth, particle.lat,
                               particle.lon]
        lon1, lat1 = (particle.lon + u1*.5*particle.dt,
                      particle.lat + v1*.5*particle.dt)
        (u2, v2) = fieldset.UV[time + .5 * particle.dt, particle.depth,
                               lat1, lon1]
        lon2, lat2 = (particle.lon + u2*.5*particle.dt,
                      particle.lat + v2*.5*particle.dt)
        (u3, v3) = fieldset.UV[time + .5 * particle.dt, particle.depth,
                               lat2, lon2]
        lon3, lat3 = (particle.lon + u3*particle.dt,
                      particle.lat + v3*particle.dt)
        (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3]
        particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
        particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt


def BrownianMotion2D(particle, fieldset, time):
    """Kernel for simple Brownian particle diffusion in zonal and meridional
    direction. Assumes that fieldset has fields Kh_zonal and Kh_meridional
    we don't want particles to jump on land and thereby beach"""
    if particle.beach == 0:
        r = 1/3.
        kh_meridional = fieldset.Kh_meridional[time, particle.depth,
                                               particle.lat, particle.lon]
        lat_p = particle.lat + ParcelsRandom.uniform(-1., 1.) * \
            math.sqrt(2*math.fabs(particle.dt)*kh_meridional/r)
        kh_zonal = fieldset.Kh_zonal[time, particle.depth,
                                     particle.lat, particle.lon]
        lon_p = particle.lon + ParcelsRandom.uniform(-1., 1.) * \
            math.sqrt(2*math.fabs(particle.dt)*kh_zonal/r)
        particle.lon = lon_p
        particle.lat = lat_p

###############################################################################
# And now the overall kernel                                                  #
###############################################################################


totalKernel = pset.Kernel(kernels.AdvectionRK4_floating) + \
    pset.Kernel(kernels.BrownianMotion2D) + \
    pset.Kernel(kernels.AntiBeachNudging) + \
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
