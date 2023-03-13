from . import coreas
import numpy as np


def make_steering_card(
    unique_identifier,
    primary_particle_type,
    energy_GeV,
    zenith_distance_deg,
    azimuth_deg,
    observation_level_asl_m,
    earth_magnetic_field_x_muT,
    earth_magnetic_field_z_muT,
):
    zd_deg = zenith_distance_deg
    az_deg = azimuth_deg
    obs_level_asl_cm = observation_level_asl_m * 1e2
    primary_particle_id = particle_id(particle_type=primary_particle_type)

    sc = "RUNNR {:d}\n".format(unique_identifier)
    sc += "EVTNR {:d}\n".format(1)
    sc += "SEED {:d} 0 0\n".format(unique_identifier + 0)
    sc += "SEED {:d} 0 0\n".format(unique_identifier + 1)
    sc += "SEED {:d} 0 0\n".format(unique_identifier + 2)
    sc += "PRMPAR {:d}\n".format(primary_particle_id)

    sc += "ERANGE {:.3E} {:.3E}\n".format(energy_GeV, energy_GeV)
    sc += "ESLOPE 0\n"
    sc += "THETAP {:.3E} {:.3E}\n".format(zenith_distance_deg, zd_deg)
    sc += "PHIP {:.3E} {:.3E}\n".format(az_deg, az_deg)

    sc += "ECUTS 3.000E-01 3.000E-01 4.010E-04 4.010E-04\n"
    sc += "ELMFLG T T\n"
    sc += "NSHOW 1\n"
    sc += "DIRECT './'\n"
    sc += "OBSLEV {:.3E}\n".format(obs_level_asl_cm)

    sc += "ECTMAP 1.000E+05\n"
    sc += "STEPFC 1.000E+00\n"
    sc += "MUMULT T\n"
    sc += "MUADDI T\n"
    sc += "PAROUT F  F\n"
    sc += "MAXPRT 1\n"
    sc += "MAGNET {:.3E} {:.3E}\n".format(
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
