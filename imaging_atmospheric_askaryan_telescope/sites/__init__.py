import os
import pkg_resources
import json_utils


def _get_resources_dir():
    return pkg_resources.resource_filename(
        "imaging_atmospheric_radio_telescope",
        os.path.join("sites", "resources"),
    )


def init(key):
    resource_path = os.path.join(_get_resources_dir(), key + ".json")
    content = json_utils.read(resource_path)
    content["key"] = key
    return content
