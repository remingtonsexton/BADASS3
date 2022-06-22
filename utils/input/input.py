import pathlib


INPUT_PARSER_PREFIX = 'read_'
SDSS_FMT = 'sdss_spec'


# To avoid circular imports, we import
# subclasses only when needed
# Note: this is the order the registry
# will look for available read functions
# TODO: should search a specific input
# parser directory and auto-import
# anything found there?
def import_custom_parsers():
    import utils.input.sdss_input
    import utils.input.ifu_input
    import utils.input.muse_input



# This registry business seems kung fu-y, but it essentially registers
# any class that inherits from BadassInput. That way custom input data
# parsers can be added easily and BadassInput will be able to find them.
# Note: The class needs to be imported in import_custom_parsers above
# so that python actually registers the subclass.
# See input_sdss.py for an example to subclass BadassInput
REGISTRY = {}

class MetaRegistry(type):
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        if name in REGISTRY:
            raise Exception('BadassInput subclass has already been defined: {name}')
        REGISTRY[cls.__name__] = cls
        return cls



class BadassInput(metaclass=MetaRegistry):

    # TODO: needs to be a prodict?
    def __init__(self, input_dict=None):
        if input_dict:
            self.__dict__.update(input_dict)


    def validate_input(self):
        return True


    def prepare_input(self, ba_ctx):
        return


    @classmethod
    def from_dict(cls, input_data):
        return cls(input_data)


    @classmethod
    def from_format(cls, input_data, options):
        import_custom_parsers()

        options = options if isinstance(options, dict) else options[0]
        fmt = options.io_options.infmt
        read_format_func = '{pre}{fmt}'.format(pre=INPUT_PARSER_PREFIX, fmt=fmt)
        for childcls in REGISTRY.values():
            if hasattr(childcls, read_format_func):
                try:
                    return getattr(childcls, read_format_func)(input_data, options)
                except:
                    raise Exception('Reading input data with format: {fmt} failed'.format(fmt=fmt))

        raise Exception('Input data format not supported: {fmt}'.format(fmt=fmt))


    @classmethod
    def from_path(cls, _path, options, filter=None):
        # TODO: implement support to filter different types
        #       of files from the supplied directory

        path = pathlib.Path(_path)
        if not path.exists():
            raise Exception('Unable to find input path: %s' % str(path))

        if path.is_file():
            return cls.from_format(path, options)

        inputs = []
        for infile in path.glob('*'):
            # TODO: support recursion into subdirs?
            if not infile.is_file():
                continue

            inputs.append(cls.from_format(infile, options))
        return inputs


    @classmethod
    def get_inputs(cls, input_data, options):
        if isinstance(input_data, list):

            if isinstance(options, list) and (len(options) != 1 and len(options) != len(input_data)):
                raise Exception('Options list must be same length as input data')

            opts = options
            if isinstance(options, dict):
                opts = [options] * len(input_data)
            elif len(options) == 1:
                opts = [options[0]] * len(input_data)

            inputs = []
            for ind, opt in zip(input_data, opts):
                inputs.extend(cls.get_inputs(ind, opt))
            return inputs

        if isinstance(input_data, dict):
            return [cls.from_dict(input_data)]

        if isinstance(input_data, pathlib.Path):
            ret = cls.from_path(input_data, options)
            return ret if isinstance(ret, list) else [ret]

        # Check if string path
        if isinstance(input_data, str):
            if pathlib.Path(input_data).exists():
                ret = cls.from_path(input_data, options)
                return ret if isinstance(ret, list) else [ret]
            # if not, could be actual data

        return [cls.from_format(input_data, options)]


    def validate_input(self):
        pass
        # TODO: implement
        # expected instance variables: spec, wave, err, fwhm, z, ebv
        # returned from prepare_*_spec functions:
        #       lam_gal, galaxy, noise, z, ebv, velscale, fwhm_gal, fit_mask_good
