from utils.input.input import BadassInput

class UserSpec(BadassInput):
	@classmethod
	def read_user_spec(cls, input_data, options):
		pass

	# see: check_user_input_spec in badass_utils.py

	# TODO: remove this class in favor of using the general
	#   BadassInput class, initiated with the from_dict function
	#   Add input validation to BadassInput class instead
