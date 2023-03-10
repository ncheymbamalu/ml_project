import sys
import logging

from src import logger


def return_error_message(error, details: sys) -> str:
    _, _, error_info = details.exc_info()
    filename: str = error_info.tb_frame.f_code.co_filename
    line_number: int = error_info.tb_lineno
    message = f"Error occurred in [{filename}] on line [{line_number}]; Error message: {str(error)}"
    return message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message: str = return_error_message(error_message, error_detail)

    def __str__(self):
        return self.error_message
