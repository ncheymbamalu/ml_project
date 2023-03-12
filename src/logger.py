import logging
import os

from datetime import datetime

filename = f"{datetime.now().strftime('%m_%d_%Y_%H_%H_%S')}.log"
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename=os.path.join("logs", filename), 
    format="[%(asctime)s] %(pathname)s %(lineno)d %(name)s - %(levelname)s - %(message)s", 
    level=logging.INFO, 
)
