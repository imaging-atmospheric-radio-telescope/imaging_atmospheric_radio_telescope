import numpy as np
import glob
import os
from . import antenna_list
from . import coreas_electric_fields


DEFAULT_TIME_BOUNDARIES = {
    "automatic_time_boundaries_s": 4e-07,
    "time_lower_boundary_s": -1,
    "time_upper_boundary_s": 1,
}


def make_steering_card(
    core_position_on_observation_level_north_m,
    core_position_on_observation_level_west_m,
    core_position_on_observation_level_asl_m,
    time_slice_duration_s,
    time_boundaries=DEFAULT_TIME_BOUNDARIES,
):
    core_north_cm = core_position_on_observation_level_north_m * 1e2
    core_west_cm = core_position_on_observation_level_west_m * 1e2
    cors_asl_cm = core_position_on_observation_level_asl_m * 1e2

    sc = "# CoREAS V1.1 by Tim Huege <tim.huege@kit.edu> with contributions "
    sc += "by Marianne Ludwig and Clancy James - parameter file\n"
    sc += "\n"
    sc += "# parameters setting up the spatial observer configuration:\n"
    sc += "\n"
    sc += "CoreCoordinateNorth = {0:.6E} ; in cm\n".format(core_north_cm)
    sc += "CoreCoordinateWest = {0:.6E} ; in cm\n".format(core_west_cm)
    sc += "CoreCoordinateVertical = {0:.6E} ; in cm\n".format(cors_asl_cm)
    sc += "\n"
    sc += "# parameters setting up the temporal observer configuration:\n"
    sc += "\n"
    sc += "TimeResolution = {0:.6E} ; in s\n".format(time_slice_duration_s)
    tb = time_boundaries
    sc += "AutomaticTimeBoundaries = {0:.6E}; ".format(
        tb["automatic_time_boundaries_s"]
    )
    sc += "0: off, x: automatic boundaries with width x in s\n"
    sc += "TimeLowerBoundary = {0:.6E} ; ".format(tb["time_lower_boundary_s"])
    sc += "in s, only if AutomaticTimeBoundaries set to 0\n"
    sc += "TimeUpperBoundary = {0:.6E} ; ".format(tb["time_upper_boundary_s"])
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
