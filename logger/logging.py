import logging
import sys

#logger = logging.getLogger('mypackage')
#ch = logging.StreamHandler(sys.stdout)
#logger.addHandler(ch)
#logger.setLevel(logging.DEBUG)

def setup_logging(log_file_path):
    """Configures logging for the entire application."""
        
    # Get the root logger
    logger = logging.getLogger() 
    logger.setLevel(logging.DEBUG)  # Set the global log level

    # Define the log format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create a file handler and set its format
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Optional: Add a stream handler for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    logger.info("Logging setup complete in main.py")