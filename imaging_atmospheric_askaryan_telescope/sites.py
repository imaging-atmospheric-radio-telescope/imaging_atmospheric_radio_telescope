def init(site_name):
    if site_name == "namibia":
        site = {
            "name": site_name,
            "observation_level_altitude": 1800,
            "earth_magnetic_field_x_muT": 12.5,
            "earth_magnetic_field_z_muT": -25.9,
        }
    elif site_name == "karlsruhe":
        site = {
            "name": site_name,
            "observation_level_altitude": 110,
            "earth_magnetic_field_x_muT": 20.4,
            "earth_magnetic_field_z_muT": 43.23,
        }
    else:
        raise AttributeError("lnb_name is not known.")

    return site
