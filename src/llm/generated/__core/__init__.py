"""
Generic functions for generated applications.
"""


def pos_args(field_names: list[str], args, kwargs) -> tuple[dict, tuple]:
    """
    Create a dictionary from positional arguments and keyword arguments.
    """
    args_dict = dict(zip(field_names, args))  # Returns {} if args is empty
    size = min(len(field_names), len(args))
    # kwargs override args if both provided
    return {**args_dict, **kwargs}, args[size:] if size > 0 else tuple()