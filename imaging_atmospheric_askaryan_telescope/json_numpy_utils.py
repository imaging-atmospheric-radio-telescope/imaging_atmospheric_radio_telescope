import numpy as np
import json


class NumPyJSONEncoder(json.JSONEncoder):
    """
    By mgilson, Software Engineer at Argo AI, 2017
    Handles numpy number types correctly
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumPyJSONEncoder, self).default(obj)
