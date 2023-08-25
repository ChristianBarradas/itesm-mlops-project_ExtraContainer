import logging

class CustomLogger:
    def __init__(self, name, log_file_path):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(module)s:%(levelname)s:%(message)s')
        
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)