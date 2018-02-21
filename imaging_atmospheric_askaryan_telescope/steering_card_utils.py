import numpy as np


"""
In CORSIKA COREAS, there must only be one shower per CORSIKA run.
"""


def make_corsika_steering_card(
    event_id=1,
    primary_particle_id=14,
    energy=100,
    zenith_distance=0.0,
    azimuth=0.0,
    observation_level_altitude=2200,
):
    zd_deg = np.rad2deg(zenith_distance)
    az_deg = np.rad2deg(azimuth)
    obs_level_cm = observation_level_altitude * 1e2

    sc = "RUNNR {:d}\n".format(event_id)
    sc += "EVTNR {:d}\n".format(1)
    sc += "SEED {:d} 0 0\n".format(event_id + 0)
    sc += "SEED {:d} 0 0\n".format(event_id + 1)
    sc += "SEED {:d} 0 0\n".format(event_id + 2)
    sc += "PRMPAR {:d}\n".format(primary_particle_id)

    sc += "ERANGE {0:.3E} {0:.3E}\n".format(energy, energy)
    sc += "ESLOPE 0\n"
    sc += "THETAP {0:.3E} {0:.3E}\n".format(zd_deg, zd_deg)
    sc += "PHIP {0:.3E} {0:.3E}\n".format(az_deg, az_deg)

    sc += "ECUTS 3.000E-01 3.000E-01 4.010E-04 4.010E-04\n"
    sc += "ELMFLG T T\n"
    sc += "NSHOW 1\n"
    sc += "DIRECT './'\n"
    sc += "OBSLEV {0:.3E}\n".format(obs_level_cm)

    sc += "ECTMAP 1.000E+05\n"
    sc += "STEPFC 1.000E+00\n"
    sc += "MUMULT T\n"
    sc += "MUADDI T\n"
    sc += "PAROUT F  F\n"
    sc += "MAXPRT 1\n"
    sc += "MAGNET 19.71 -14.18\n"
    sc += "LONGI T   5.  T  T\n"
    sc += "RADNKG 5.000E+05\n"
    sc += "DATBAS F\n"

    sc += "EXIT\n"
    return sc


DEFAULT_COREAS_TIME_BOUNDARIES = {
    'automatic_time_boundaries':4e-07,
    'time_lower_boundary':-1,
    'time_upper_boundary':1,
}


def make_coreas_steering_card(
    core_position_on_observation_level_north=0.0,
    core_position_on_observation_level_west=0.0,
    observation_level_altitude=2200,
    time_slice_duration=2e-10,
    time_boundaries=DEFAULT_COREAS_TIME_BOUNDARIES,
):
    core_north_cm = core_position_on_observation_level_north * 1e2
    core_west_cm = core_position_on_observation_level_west * 1e2
    obs_level_cm = observation_level_altitude * 1e2

    sc = "# CoREAS V1.1 by Tim Huege <tim.huege@kit.edu> with contributions "
    sc += "by Marianne Ludwig and Clancy James - parameter file\n"
    sc += "\n"
    sc += "# parameters setting up the spatial observer configuration:\n"
    sc += "\n"
    sc += "CoreCoordinateNorth = {0:.6E} ; in cm\n".format(core_north_cm)
    sc += "CoreCoordinateWest = {0:.6E} ; in cm\n".format(core_west_cm)
    sc += "CoreCoordinateVertical = {0:.6E} ; in cm\n".format(obs_level_cm)
    sc += "\n"
    sc += "# parameters setting up the temporal observer configuration:\n"
    sc += "\n"
    sc += "TimeResolution = {0:.6E} ; in s\n".format(time_slice_duration)
    tb = time_boundaries
    sc += "AutomaticTimeBoundaries = {0:.6E}; ".format(
        tb['automatic_time_boundaries'])
    sc += "0: off, x: automatic boundaries with width x in s\n"
    sc += "TimeLowerBoundary = {0:.6E} ; ".format(tb['time_lower_boundary'])
    sc += "in s, only if AutomaticTimeBoundaries set to 0\n"
    sc += "TimeUpperBoundary = {0:.6E} ; ".format(tb['time_upper_boundary'])
    sc += "in s, only if AutomaticTimeBoundaries set to 0\n"
    sc += "ResolutionReductionScale = 0 ; "
    sc += "0: off, x: decrease time resolution linearly every x cm in radius\n"
    sc += "\n"
    sc += "# parameters setting up the simulation functionality:\n"
    sc += "GroundLevelRefractiveIndex = 1.000292 ; "
    sc += "specify refractive index at 0 m asl\n"
    sc += "\n"
    sc += "# event information for Offline simulations:\n"
    sc += "\n"
    sc += "EventNumber = -1\n"
    sc += "RunNumber = -1\n"
    sc += "GPSSecs = 0\n"
    sc += "CoreEastingOffline = 0 ; in meters\n"
    sc += "CoreNorthingOffline = 0 ; in meters\n"
    sc += "CoreVerticalOffline = 0 ; in meters\n"
    sc += "RotationAngleForMagfieldDeclination = 0     ; in degrees\n"
    sc += "Comment = \n"
    sc += "\n"
    sc += "# event information for your convenience and backwards "
    sc += "compatibility with other software, these values are not used as "
    sc += "input parameters for the simulation:\n"
    sc += "\n"
    return sc
