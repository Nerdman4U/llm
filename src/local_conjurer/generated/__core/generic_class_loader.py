"""
generic_class_loader.py
"""
import importlib
from loguru_wrapper import loguru


def load_and_validate_generated_class(module_path: str, class_name: str, logger=None):
    """
    Dynamically load and validate a generated class with comprehensive error handling.

    This function attempts to import a generated class module and validate that the class
    has the expected interface. It's designed to gracefully handle missing generated files
    (which is normal during development) and provide detailed logging for debugging.

    Args:
        module_path (str): Full Python module path to the generated class module.
                          Example: "gim_metaproject.generated.generator.project.manager.emacs_org_mode.emacs_manager_generator"
        class_name (str): Name of the class to load from the module.
                         Example: "EmacsManagerGenerator"
        logger (Optional): Logger instance to use. If None, uses the default generated logger.

    Returns:
        tuple[module, type] | tuple[None, None]: A tuple containing:
            - module: The imported module object if successful, None if failed
            - cls: The loaded class object if successful, None if failed

    Raises:
        No exceptions are raised. All errors are logged and handled gracefully.

    Validation Checks:
        - Module can be imported successfully
        - Class exists in the module
        - Class is actually a class type (not a variable or function)
        - Class has required methods: 'initialization_params', '__init__'

    Logging Behavior:
        - DEBUG: Module import attempts
        - INFO: Successful class loading
        - WARNING: Class not found or invalid type
        - ERROR: Missing required methods
        - CRITICAL: Import errors and unexpected exceptions

    Example:
        >>> # Load a generated manager class
        >>> module, cls = load_and_validate_generated_class(
        ...     "gim_metaproject.generated.generator.project.manager.emacs_org_mode.emacs_manager_generator",
        ...     "EmacsManagerGenerator"
        ... )
        >>>
        >>> if cls:
        ...     # Use the generated class
        ...     instance = cls(*args, **kwargs)
        ... else:
        ...     # Fall back to manual implementation
        ...     instance = None

    Note:
        This function is designed to be used in the metaproject's conditional architecture
        where generated classes are optional enhancements. ImportError is expected and
        normal when generated files don't exist yet.
    """
    logger = loguru()

    try:
        # Import the module
        logger.debug(f"Attempting to import generated module: {module_path}")
        module = importlib.import_module(module_path)

        # Get the class
        cls = getattr(module, class_name, None)
        if cls is None:
            logger.warning(f"Class '{class_name}' not found in module '{module_path}'")
            return None, None

        # Validate it's a class
        if not isinstance(cls, type):
            logger.warning(f"'{class_name}' is not a class in module '{module_path}'")
            return None, None

        # Validate expected interface
        expected_methods = ['initialization_params', '__init__']
        for method_name in expected_methods:
            if not hasattr(cls, method_name):
                logger.error(
                    f"Generated class missing expected method: {method_name}")
                return None, None

        logger.info(f"Successfully loaded generated class: {class_name}")
        return module, cls

    except ImportError as e:
        logger.critical(f"Generated module not found (this is normal): {e}")
        return None, None
    except Exception as e:  # pylint: disable=broad-except
        logger.critical(f"Unexpected error loading generated class: {e}")
        return None, None
