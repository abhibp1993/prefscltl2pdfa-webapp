import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from webapp import server as app
from loguru import logger
from pathlib import Path

CUR_DIR = Path(__file__).resolve().parent
logger.add(CUR_DIR / "out" / "app.log", rotation="100 MB", level="DEBUG")
# logger.info(sys.path)
# logger.info(app.get_asset_url("app.log"))

if __name__ == "__main__":
    app.run(port=5081)
