import numpy as np

from . import coreas
from . import build


TOP_OF_ATMOSPHERE_ALTITUDE_M = 115e3


def make_steering_card(
    unique_identifier,
    primary_particle_key,
    energy_GeV,
    zenith_rad,
    azimuth_rad,
    observation_level_asl_m,
    earth_magnetic_field_x_muT,
    earth_magnetic_field_z_muT,
):
    obs_level_asl_cm = observation_level_asl_m * 1e2
    primary_particle_id = particle_key_to_corsika_id(
        particle_key=primary_particle_key
    )
    r2d = np.rad2deg

    sc = "RUNNR {:d}\n".format(unique_identifier)
    sc += "EVTNR {:d}\n".format(1)
    sc += "SEED {:d} 0 0\n".format(unique_identifier + 0)
    sc += "SEED {:d} 0 0\n".format(unique_identifier + 1)
    sc += "SEED {:d} 0 0\n".format(unique_identifier + 2)
    sc += "PRMPAR {:d}\n".format(primary_particle_id)

    sc += "ERANGE {:.E} {:.E}\n".format(energy_GeV, energy_GeV)
    sc += "ESLOPE 0\n"
    sc += "THETAP {:.E} {:.E}\n".format(r2d(zenith_rad), r2d(zenith_rad))
    sc += "PHIP {:.E} {:.E}\n".format(r2d(azimuth_rad), r2d(azimuth_rad))

    sc += "ECUTS 3.000E-01 3.000E-01 4.010E-04 4.010E-04\n"
    sc += "ELMFLG T T\n"
    sc += "NSHOW 1\n"
    sc += "DIRECT './'\n"
    sc += "OBSLEV {:.E}\n".format(obs_level_asl_cm)

    sc += "ECTMAP 1.000E+05\n"
    sc += "STEPFC 1.000E+00\n"
    sc += "MUMULT T\n"
    sc += "MUADDI T\n"
    sc += "PAROUT F  F\n"
    sc += "MAXPRT 1\n"
    sc += "MAGNET {:.E} {:.E}\n".format(
        earth_magnetic_field_x_muT,
        earth_magnetic_field_z_muT,
    )
    sc += "LONGI T   5.  T  T\n"
    sc += "RADNKG 5.000E+05\n"
    sc += "DATBAS F\n"

    sc += "EXIT\n"
    return sc


def particle_key_to_corsika_id(particle_key):
    m = {
        "gamma": 1,
        "electron": 3,
        "proton": 14,
        "helium": 402,
    }
    return m[particle_key]
