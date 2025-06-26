"""
Shared test methods for Python Application Generator 3.10 Basic Structure
"""
import pathlib
from unittest.mock import patch

import tests.generated.shared_test_methods as generated


def logger():
    """Return logger for the shared test methods"""
    return generated.logger()


def getenv(test, key):
    """Getenv for mocked environment"""
    if key == "GIM_APPLICATIONS_ROOTPATH":
        return test.workspace

    raise KeyError(f"Environment variable {key} not found.")

def set_up(test):
    """Set up the test environment"""
    generated.set_up(test)
    patcher = patch("os.getenv", side_effect=lambda key: getenv(test, key))
    patcher.start()
    test.addCleanup(patcher.stop)


def tear_down(test, to_be_removed):
    """Tear down the test environment"""
    if not to_be_removed:
        to_be_removed = []

    whitelist = [
        "workspace/testi2",
        "workspace/testi2/tests/extension/test_my_class3.py",
        "workspace/testi2/tests/generated/test_my_class3.py",
    ]

    # if any whitelisted matches blacklisted path true is returned and filter removes it
    whitelist: list = [pathlib.Path(filepath) for filepath in whitelist]
    skipped: list = list(
        filter(
            lambda blacklisted: any(
                map(
                    lambda whitelisted: whitelisted.is_relative_to(blacklisted),
                    whitelist,
                )
            ),
            to_be_removed,
        )
    )
    result = set(to_be_removed) - set(skipped)
    generated.tear_down(test, result)