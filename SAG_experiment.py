#!/usr/bin/env python
# coding: utf-8
from parcels import FieldSet, ParticleSet, AdvectionRK4, JITParticle
from parcels import ErrorCode, DiffusionUniformKh, Field
from datetime import timedelta
import datetime
import numpy as np
import pickle
import sys

n_points = 10000  # particles per sampling site
n_days = 1  # 22*30  # number of days to simulate
K_bar = 10  # diffusion value
n_site = 13
stored_dt = 1  # hours
loc = sys.argv[1]
# The file go from:
# 23 oct 2018 - 23 nov 2018
# 23 nov 2018 - 23 dic 2018
# 23 dic 2018 - 23 jan 2019

# data = '../data/mercatorpsy4v3r1_gl12_mean_20180101_R20180110.nc'
data = 'data/mercatorpsy4v3r1_gl12_mean_20180101_R20180110.nc'
output_path = f'data/source_{loc}_release.nc'
# data = '/data/oceanparcels/input_data/CMEMS/GLOBAL_ANALYSIS_FORECAST_PHY_001_024/*.nc'  # gemini
# output_path = f'/scratch/cpierard/source_{loc}_release.nc'

# time range 2018-01-01 to 2019-11-27
filesnames = {'U': data,
              'V': data}

variables = {'U': 'uo',
             'V': 'vo'}  # Use utotal

dimensions = {'lat': 'latitude',
              'lon': 'longitude',
              'time': 'time'}
indices = {'lat': range(1, 900), 'lon': range(1284, 2460)}


def delete_particle(particle, fieldset, time, indices=indices):
    particle.delete()


fieldset = FieldSet.from_netcdf(filesnames, variables, dimensions,
                                allow_time_extrapolation=True, indices=indices)

# Diffusion
size2D = (fieldset.U.grid.ydim, fieldset.U.grid.xdim)

fieldset.add_field(Field('Kh_zonal', data=K_bar * np.ones(size2D),
                         lon=fieldset.U.grid.lon, lat=fieldset.U.grid.lat,
                         mesh='spherical'))
fieldset.add_field(Field('Kh_meridional', data=K_bar * np.ones(size2D),
                         lon=fieldset.U.grid.lon, lat=fieldset.U.grid.lat,
                         mesh='spherical'))

# Opening file with positions and sampling dates.
with open('river_sources.pkl', 'rb') as infile:
    river_sources = pickle.load(infile)

np.random.seed(2)  # to repeat experiment in the same conditions
# Create the cluster of particles around the sampling site
# with a radius of 1/24 deg (?).
time = datetime.datetime.strptime('2018-01-01 12:00:00', '%Y-%m-%d %H:%M:%S')

lon_cluster = [river_sources[loc][1]]*n_points
lat_cluster = [river_sources[loc][0]]*n_points

lon_cluster = np.array(lon_cluster)+(np.random.random(len(lon_cluster))-0.5)/24
lat_cluster = np.array(lat_cluster)+(np.random.random(len(lat_cluster))-0.5)/24
date_cluster = np.repeat(time, n_points)

# creating the Particle set
pset = ParticleSet.from_list(fieldset=fieldset,
                             pclass=JITParticle,
                             lon=lon_cluster,
                             lat=lat_cluster,
                             time=date_cluster)

# Output file
output_file = pset.ParticleFile(
    name=output_path,
    outputdt=timedelta(hours=stored_dt))

# Execute!
pset.execute(pset.Kernel(AdvectionRK4) + DiffusionUniformKh,
             runtime=timedelta(days=n_days),
             dt=timedelta(hours=1),
             output_file=output_file,
             recovery={ErrorCode.ErrorOutOfBounds: delete_particle})
output_file.close()
