def init(key):
    if key == "namibia":
        site = {
            "key": key,
            "observation_level_asl_m": 1800,
            "earth_magnetic_field_x_muT": 12.5,
            "earth_magnetic_field_z_muT": -25.9,
            "name": "Khoma Highlands, H.E.S.S. site, Namibia",
        }
    elif key == "karlsruhe":
        site = {
            "key": key,
            "observation_level_asl_m": 110,
            "earth_magnetic_field_x_muT": 20.4,
            "earth_magnetic_field_z_muT": 43.23,
            "name": "Campus of the Karlsruhe Institute of Technology, Germany",
        }
    else:
        raise AttributeError(f"Site key '{key:s}' is not known.")

    return site
