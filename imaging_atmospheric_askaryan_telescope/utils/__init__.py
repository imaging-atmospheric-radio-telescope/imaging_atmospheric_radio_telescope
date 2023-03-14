from . import tarstream


def strip_dict(obj, strip):
    out = {}
    for key in obj:
        if key != strip:
            item = obj[key]
            if isinstance(item, dict):
                out[key] = strip_dict(obj=item, strip=strip)
            else:
                out[key] = item
    return out
