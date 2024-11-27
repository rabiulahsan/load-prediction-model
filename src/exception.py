import sys
import os
import traceback

from src.logger import logging


def error_message_details() -> str:
    """Extracts the current exception details and returns a formatted traceback string."""
    return traceback.format_exc()

class CustomException(Exception):
    def __init__(self, error_message):
        super().__init__(error_message)
        # Capture the formatted traceback as the error message
        self.error_message = f"{error_message}\n{error_message_details()}"
        logging.error(self.error_message)  # Log the error details
    
    def __str__(self):
        return self.error_message


# for testing 
# if __name__ == '__main__':
#     try:
#         result = 10 / 0  # Division by zero to trigger an exception
#     except Exception as e:
#         logging.info("Divided by zero.")
#         raise CustomException(e)
