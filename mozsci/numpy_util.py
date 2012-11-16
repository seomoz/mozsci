
import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):
    """A JSON encoder for numpy arrays
    Use like json.dumps(data, cls=NumpyEncoder)"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def numpy_decoder(dct):
    """Decodes numpy arrays stored as values in a json dictionary
    Use like json.loads(j, object_hook=numpy_decoder)"""
    for k in dct.keys():
        if isinstance(dct[k], list):
            try:
                dct[k] = np.asarray(dct[k])
            except ValueError:
                pass   # can't convert to numpy array so leave as is
    return dct


def load_json_to_numpy(jsonfile):
    """Loads the data in the jsonfile using numpy_decoder to convert
    to numpy arrays.  Returns the decoded data"""
    return json.load(open(jsonfile, 'r'),
                            object_hook=numpy_decoder)


