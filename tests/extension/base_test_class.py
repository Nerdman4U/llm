"""
Base test class for Python unit tests.
"""
import unittest
import pathlib


class BaseTestClass(unittest.TestCase):
    """
    Base class for all test cases in the extension.

    Introduces class attributes for common paths and fixtures.
    """
    fixtures: pathlib.Path
    workspace: pathlib.Path
    install_path: pathlib.Path
    projects_path: pathlib.Path