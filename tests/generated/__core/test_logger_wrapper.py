"""
Generated unittests for generated utility methods.
"""

import unittest
from unittest.mock import patch, MagicMock
from loguru import logger as loguru_logger

from local_conjurer.generated.__core import logger_wrapper


class TestLoggerWrapper(unittest.TestCase):  # pylint: disable=too-many-public-methods
    """Test cases for generated utility methods."""

    def test_logger_returns_loguru_logger(self):
        """Test that logger() returns the loguru logger instance."""
        result = logger_wrapper.logger()
        self.assertEqual(result, loguru_logger)

    def test_lazy_logger_returns_opt_lazy(self):
        """Test that lazy_logger() returns loguru logger with lazy option."""
        with patch.object(loguru_logger, 'opt') as mock_opt:
            mock_opt.return_value = MagicMock()
            result = logger_wrapper.lazy_logger()
            mock_opt.assert_called_once_with(lazy=True)
            self.assertEqual(result, mock_opt.return_value)

    def test_make_lambda_returns_callable(self):
        """Test that make_lambda returns a callable that returns the given value."""
        test_value = "test_value"
        lambda_func = logger_wrapper.make_lambda(test_value)

        self.assertTrue(callable(lambda_func))
        self.assertEqual(lambda_func(), test_value)

    def test_make_lambda_with_different_types(self):
        """Test make_lambda with different value types."""
        test_cases = [
            42,
            "string",
            [1, 2, 3],
            {"key": "value"},
            None
        ]

        for value in test_cases:
            with self.subTest(value=value):
                lambda_func = logger_wrapper.make_lambda(value)
                self.assertEqual(lambda_func(), value)

    def test_transform_args_with_callables(self):
        """Test transform_args preserves callable arguments."""
        def test_func():
            return "test"

        args = [test_func, lambda: "lambda"]
        result = logger_wrapper.transform_args(args)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], test_func)
        self.assertTrue(callable(result[1]))

    def test_transform_args_with_non_callables(self):
        """Test transform_args wraps non-callable arguments in lambdas."""
        args = ["string", 42, None]
        result = logger_wrapper.transform_args(args)

        self.assertEqual(len(result), 3)
        for item in result:
            self.assertTrue(callable(item))

        self.assertEqual(result[0](), "string")
        self.assertEqual(result[1](), 42)
        self.assertEqual(result[2](), None)

    def test_transform_args_mixed(self):
        """Test transform_args with mixed callable and non-callable arguments."""
        def test_func():
            return "callable"

        args = [test_func, "string", 42]
        result = logger_wrapper.transform_args(args)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], test_func)
        self.assertTrue(callable(result[1]))
        self.assertTrue(callable(result[2]))
        self.assertEqual(result[1](), "string")
        self.assertEqual(result[2](), 42)

    def test_transform_args_empty_list(self):
        """Test transform_args with empty list."""
        result = logger_wrapper.transform_args([])
        self.assertEqual(result, [])

    @patch('locshapython.generated.__core.logger_wrapper.do_log')
    def test_debug_calls_do_log(self, mock_do_log):
        """Test debug function calls do_log with correct method_name."""
        logger_wrapper.debug("test message", "arg1")
        mock_do_log.assert_called_once_with("test message", "arg1", method_name="debug")

    @patch('locshapython.generated.__core.logger_wrapper.do_log')
    def test_info_calls_do_log(self, mock_do_log):
        """Test info function calls do_log with correct method_name."""
        logger_wrapper.info("test message", "arg1")
        mock_do_log.assert_called_once_with("test message", "arg1", method_name="info")

    @patch('locshapython.generated.__core.logger_wrapper.do_log')
    def test_success_calls_do_log(self, mock_do_log):
        """Test success function calls do_log with correct method_name."""
        logger_wrapper.success("test message", "arg1")
        mock_do_log.assert_called_once_with(
            "test message", "arg1", method_name="success")

    @patch('locshapython.generated.__core.logger_wrapper.do_log')
    def test_warning_calls_do_log(self, mock_do_log):
        """Test warning function calls do_log with correct method_name."""
        logger_wrapper.warning("test message", "arg1")
        mock_do_log.assert_called_once_with(
            "test message", "arg1", method_name="warning")

    @patch('locshapython.generated.__core.logger_wrapper.do_log')
    def test_error_calls_do_log(self, mock_do_log):
        """Test error function calls do_log with correct method_name."""
        logger_wrapper.error("test message", "arg1")
        mock_do_log.assert_called_once_with("test message", "arg1", method_name="error")

    @patch('locshapython.generated.__core.logger_wrapper.do_log')
    def test_critical_calls_do_log(self, mock_do_log):
        """Test critical function calls do_log with correct method_name."""
        logger_wrapper.critical("test message", "arg1")
        mock_do_log.assert_called_once_with(
            "test message", "arg1", method_name="critical")

    @patch('locshapython.generated.__core.logger_wrapper.lazy_logger')
    def test_do_log_basic_functionality(self, mock_lazy_logger):
        """Test do_log basic functionality."""
        mock_logger = MagicMock()
        mock_debug = MagicMock()
        mock_logger.debug = mock_debug
        mock_lazy_logger.return_value = mock_logger

        logger_wrapper.do_log("test message", method_name="debug")

        mock_debug.assert_called_once_with("test message")

    @patch('locshapython.generated.__core.logger_wrapper.lazy_logger')
    def test_do_log_with_args(self, mock_lazy_logger):
        """Test do_log with additional arguments."""
        mock_logger = MagicMock()
        mock_debug = MagicMock()
        mock_logger.debug = mock_debug
        mock_lazy_logger.return_value = mock_logger

        logger_wrapper.do_log("test message {}", "arg1", method_name="debug")

        # Should call debug with message and transformed args
        mock_debug.assert_called_once()
        args, kwargs = mock_debug.call_args  # pylint: disable=unused-variable
        self.assertEqual(args[0], "test message {}")
        self.assertEqual(len(args), 2)
        self.assertTrue(callable(args[1]))  # arg should be wrapped in lambda

    @patch('locshapython.generated.__core.logger_wrapper.lazy_logger')
    def test_do_log_percent_s_replacement(self, mock_lazy_logger):
        """Test do_log replaces %s with {} in message."""
        mock_logger = MagicMock()
        mock_debug = MagicMock()
        mock_logger.debug = mock_debug
        mock_lazy_logger.return_value = mock_logger

        logger_wrapper.do_log("test message %s and %s", "arg1",
                              "arg2", method_name="debug")

        args, kwargs = mock_debug.call_args  # pylint: disable=unused-variable
        self.assertEqual(args[0], "test message {} and {}")

    def test_do_log_no_message_raises_error(self):
        """Test do_log raises ValueError when no message provided."""
        with self.assertRaises(ValueError) as context:
            logger_wrapper.do_log()

        self.assertEqual(str(context.exception), "Message must be provided for logging")

    def test_do_log_non_string_message_raises_error(self):
        """Test do_log raises TypeError when message is not a string."""
        with self.assertRaises(TypeError) as context:
            logger_wrapper.do_log(123, method_name="debug")

        self.assertEqual(str(context.exception), "Message must be a string")

    def test_do_log_invalid_method_raises_error(self):
        """Test do_log raises ValueError for invalid method name."""

        # Simulate getattr returning None for invalid method
        with self.assertRaises(ValueError) as context:
            logger_wrapper.do_log("test", method_name="invalid_method")

        self.assertEqual(str(context.exception),
                         "Method 'invalid_method' is not a valid loguru method")

    @patch('locshapython.generated.__core.logger_wrapper.lazy_logger')
    def test_do_log_message_from_kwargs(self, mock_lazy_logger):
        """Test do_log can get message from kwargs."""
        mock_logger = MagicMock()
        mock_debug = MagicMock()
        mock_logger.debug = mock_debug
        mock_lazy_logger.return_value = mock_logger

        logger_wrapper.do_log(message="test message", method_name="debug")

        mock_debug.assert_called_once_with("test message")

    @patch('locshapython.generated.__core.logger_wrapper.lazy_logger')
    def test_do_log_different_log_levels(self, mock_lazy_logger):
        """Test do_log works with different log levels."""
        mock_logger = MagicMock()
        mock_lazy_logger.return_value = mock_logger

        log_levels = ["debug", "info", "success", "warning", "error", "critical"]

        for level in log_levels:
            with self.subTest(level=level):
                mock_method = MagicMock()
                setattr(mock_logger, level, mock_method)

                logger_wrapper.do_log("test message", method_name=level)

                mock_method.assert_called_once_with("test message")
                mock_method.reset_mock()


if __name__ == '__main__':
    unittest.main()