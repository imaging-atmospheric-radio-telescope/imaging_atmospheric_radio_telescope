import os
import json_utils


def get_configfile_path(programname="corsika_coreas.json"):
    return os.path.join(os.path.expanduser("~"), "." + programname)


def read():
    configfile_path = get_configfile_path()
    if not os.path.exists(configfile_path):
        write(config=default(build_dir="build"))

    with open(configfile_path, "rt") as f:
        config = json_utils.loads(f.read())
    return config


def write(config):
    with open(get_configfile_path(), "wt") as f:
        f.write(json_utils.dumps(config, indent=4))


def default(build_dir):
    out = {
        "vanilla": os.path.join(
            os.path.abspath(build_dir),
            "corsika-77100",
            "run",
            "corsika77100Linux_QGSII_urqmd_coreas",
        ),
    }
    return out
