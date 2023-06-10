import json


def read_params(filename):
    with open(filename, 'r') as f:
        data = f.read()
    params = json.loads(data)
    return Hps(**params)


class Hps:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = Hps(**v)
            self[k] = v

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)
