import numpy as np


def dumps(positions_asl_m, prefix=""):
    """
    Parameters
    ----------
    positions_asl_m : array
        Positions of antennas above sea level (asl).
    prefix : str, default=""
        Will be put infornt of the numerical index to form the antenna's name.

    Returns
    -------
    text : str
        The content of the CoREAS antenna list text file.
    """
    template_line = "AntennaPosition = {x:2f}\t{y:2f}\t{z:2f}\t "
    template_line += "{prefix:s}{antenna_idx:06d}\n"
    antenna_list = ""
    for i in range(positions_asl_m.shape[0]):
        antenna_list += template_line.format(
            x=positions_asl_m[i, 0] * 1e2,
            y=positions_asl_m[i, 1] * 1e2,
            z=positions_asl_m[i, 2] * 1e2,
            prefix=prefix,
            antenna_idx=i,
        )
    return antenna_list


def loads(text, prefix=None):
    """
    Parameters
    ----------
    text : str
        The content of the CoREAS antenna list text file.
    prefix : str, default=None
        Will only consider antenna positions which have a name starting with
        'prefix'.

    Returns
    -------
    positions_asl_m : array shape(N x 3)
        The positions of the N antennas
    """
    positions_asl_m = []
    for line in str.splitlines(text):

        if line.startswith("AntennaPosition = "):
            part = line.replace("AntennaPosition = ", "")
            tokens = part.split("\t")

            x_cm = float(tokens[0])
            y_cm = float(tokens[1])
            z_cm = float(tokens[2])
            name = tokens[3]

            position_asl_m = 1e-2 * np.array([x_cm, y_cm, z_cm])

            if prefix is None:
                positions_asl_m.append(position_asl_m)
            else:
                if name.startswith(prefix):
                    positions_asl_m.append(position_asl_m)

    if len(positions_asl_m) == 0:
        return np.zeros(shape=(0, 3), dtype=float)
    else:
        return np.asarray(positions_asl_m)
