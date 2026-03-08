from pathlib import Path
import yaml

CONFIG_PATH = Path("eval/config.yaml")
CONFIG = yaml.safe_load(CONFIG_PATH.read_text())
