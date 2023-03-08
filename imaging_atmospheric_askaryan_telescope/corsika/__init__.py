from . import coreas
import numpy as np


def make_steering_card(
    event_id,
    primary_particle_type,
    energy,
    zenith_distance,
    azimuth,
    observation_level_altitude,
    earth_magnetic_field_x_muT,
    earth_magnetic_field_z_muT,
):
    zd_deg = np.rad2deg(zenith_distance)
    az_deg = np.rad2deg(azimuth)
    obs_level_cm = observation_level_altitude * 1e2
    primary_particle_id = particle_id(particle_type=primary_particle_type)

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
    sc += "MAGNET {0:.3E} {0:.3E}\n".format(
        earth_magnetic_field_x_muT, earth_magnetic_field_z_muT,
    )
    sc += "LONGI T   5.  T  T\n"
    sc += "RADNKG 5.000E+05\n"
    sc += "DATBAS F\n"

    sc += "EXIT\n"
    return sc


def particle_id(particle_type):
    m = {
        "gamma": 1,
        "proton": 14,
        "helium": 402,
    }
    return m[particle_type]
