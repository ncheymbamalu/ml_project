import logging
import os

from datetime import datetime

filename = f"{datetime.now().strftime('%m_%d_%Y_%H_%H_%S')}.log"
filepath = os.path.join(os.getcwd(), "logs")
os.makedirs(filepath, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(filepath, filename), 
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s", 
    level=logging.INFO, 
)
