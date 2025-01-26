# logger.py
import logging
from pathlib import Path
from config import PATHS


class Logger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler(PATHS["logs"] / f"{name}.log", encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
    
    def error(self, msg: str) -> None:
        self.logger.error(msg)
    
    def info(self, msg: str) -> None:
        self.logger.info(msg)


logger = Logger("app")
