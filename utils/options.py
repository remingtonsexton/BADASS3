import importlib
import json
import pathlib
import prodict

from utils.schema import DefaultValidator, DEFAULT_OPTIONS_SCHEMA


class BadassOptions(prodict.Prodict):

    @classmethod
    def from_dict(cls, input_dict):
        # Override Prodict.from_dict to normalize and validate input
        v = DefaultValidator(DEFAULT_OPTIONS_SCHEMA)

        # Update dict with default values if needed
        input_dict = v.normalized(input_dict)
        if not v.validate(input_dict):
            raise Exception('Options validation failed: %s' % v.errors)

        return super().from_dict(input_dict)

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


    # Custom file type parsers
    # Note: each parser should parse options to a dict and use
    #   BadassOptions.from_dict to initialize, allowing for
    #   option normalization and validation

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

