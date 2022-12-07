from . import coreas
import numpy as np


def make_steering_card(
    event_id=1,
    primary_particle_id=14,
    energy=100,
    zenith_distance=0.0,
    azimuth=0.0,
    observation_level_altitude=2200,
    earth_magnetic_field_x_muT=12.5,
    earth_magnetic_field_z_muT=-25.9,
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
    sc += "MAGNET {0:.3E} {0:.3E}\n".format(
        earth_magnetic_field_x_muT, earth_magnetic_field_z_muT,
    )
    sc += "LONGI T   5.  T  T\n"
    sc += "RADNKG 5.000E+05\n"
    sc += "DATBAS F\n"

    sc += "EXIT\n"
    return sc
