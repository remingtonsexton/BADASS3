# TODO: default values are retrieved from default json file

class Verifier:
    def verify(self, opts):
        return None


# Verifier to check the value of a single option
class ValueVerifier(Verifier):
    def __init__(self, keys, types, conds, default, err_msg):
        self.keys = keys
        self.types = types
        self.conds = conds
        self.default = default
        self.err_msg = err_msg

    def verify(self, opts):
        ek = self.keys[-1]

        # Get the immediate parent dictionary
        # of the last key in case it needs
        # to be set later
        topts = opts
        for k in self.keys[:-1]:
            if not k in topts:
                raise Exception('Option \'{k}\' not provided in parent option'.format(k=k))
            topts = topts[k]

        if ek not in topts:
            topts[ek] = self.default

        if self.types and not isinstance(topts[ek], self.types):
            raise ValueError(self.err_msg)

        if not all(f(topts[ek]) for f in self.conds):
            raise ValueError(self.err_msg)

        return None


# Verifier to check the validity of options
# across multiple options
class CrossOptionVerifier(Verifier):
    def __init__(self, conds, err_msg):
        self.conds = conds
        self.err_msg = err_msg

    def verify(self, opts):
        if not all(f(opts) for f in self.conds):
            raise ValueError(self.err_msg)
