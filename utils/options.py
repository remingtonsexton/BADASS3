import importlib
import json
import pathlib
import prodict

from utils.verify.verify_default import DefaultVerifySet

# TODO: validation, all expected options are supplied

class BadassOptions(prodict.Prodict):
    @classmethod
    def from_file(cls, _filepath):
        filepath = pathlib.Path(_filepath)
        if not filepath.exists():
            raise Exception('Unable to find options file: %s' % str(filepath))

        ext = filepath.suffix[1:]
        parse_func_name = 'parse_%s' % ext
        if not hasattr(cls, parse_func_name):
            raise Exception('Unsupported option file type: %s' % ext)

        return getattr(cls, parse_func_name)(filepath)

    @classmethod
    def parse_json(cls, filepath):
        return cls.from_dict(json.load(filepath.open()))

    @classmethod
    def parse_py(cls, filepath):
        # TODO: better way to import or parse module
        #   + could add filepath's parent directory
        #   + to PYTHONPATH, import the module name
        #   + then remove the directory?
        o = importlib.import_module(str(filepath.relative_to(pathlib.Path.cwd())).rstrip(filepath.suffix).replace('/', '.'))
        return cls.from_dict({k:getattr(o, k) for k in dir(o) if not k[:2] == '__'})

    @classmethod
    def get_options(cls, options_data):
        if isinstance(options_data, list):
            return [cls.get_options(o) for o in options_data]

        if isinstance(options_data, dict):
            return [cls.from_dict(options_data)]

        if isinstance(options_data, pathlib.Path) or isinstance(options_data, str):
            return [cls.from_file(options_data)]

        return []

    def verify(self, verify_set=DefaultVerifySet):
        verify_set(self).verify()
