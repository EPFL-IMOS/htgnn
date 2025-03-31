import os
from dotenv import load_dotenv

load_dotenv()

NEPTUNE_PROJECT_NAME_BEARING = os.getenv('NEPTUNE_PROJECT_NAME_BEARING')
NEPTUNE_PROJECT_NAME_BRIDGE = os.getenv('NEPTUNE_PROJECT_NAME_BRIDGE')
NEPTUNE_API_TOKEN = os.getenv('NEPTUNE_API_TOKEN')
