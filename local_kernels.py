from parcels import ParcelsRandom
from parcels import AdvectionRK4, DiffusionUniformKh
import math


def delete_particle(particle, fieldset, time):  # indices=indices):
    particle.delete()


def beach(particle, fieldset, time):

    if particle.beach == 0:
        dist = fieldset.distance2shore[time, particle.depth, particle.lat,
                                       particle.lon]
        if dist < 10:
            beach_prob = math.exp(-particle.dt/(particle.coastPar*86400.))
            if ParcelsRandom.random(0., 1.) > beach_prob:
                particle.beach = 1
    # Now the part where we build in the resuspension
    elif particle.beach == 1:
        resus_prob = math.exp(-particle.dt/(particle.resus_t*86400.))
        if ParcelsRandom.random(0., 1.) > resus_prob:
            particle.beach = 0
    # Update the age of the particle
    particle.age += particle.dt


def AntiBeachNudging(particle, fieldset, time):
    """
    The nudging current is 1 m s^-1, which ought to be sufficient to overpower
    any coastal current (I hope) and push our particle back out to sea so as to
    not get stuck

    update 11/03/2020: Following tests and discussions with Cleo, the nudging
    current will now kick in starting at 500m from the coast, since otherwise
    the particles tended to get stuck if we used the velocity treshhold.
    """

    particle.distance = fieldset.distance2shore[time, particle.depth,
                                                particle.lat, particle.lon]

    if fieldset.distance2shore[time, particle.depth,
                               particle.lat, particle.lon] < 0.5:
        borUab = fieldset.borU[time, particle.depth, particle.lat,
                               particle.lon]
        borVab = fieldset.borV[time, particle.depth, particle.lat,
                               particle.lon]
        particle.lon += borUab*particle.dt
        particle.lat += borVab*particle.dt


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
        dWx = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))
        dWy = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))

        bx = math.sqrt(2 * fieldset.Kh_zonal[particle])
        by = math.sqrt(2 * fieldset.Kh_meridional[particle])

        particle.lon += bx * dWx
        particle.lat += by * dWy
