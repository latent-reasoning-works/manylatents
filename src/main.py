import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

PROJECT_NAME = Path(__file__).parent.name
REPO_ROOTDIR = Path(__file__).parent.parent
logger = logging.getLogger(__name__)

@hydra.main(
    config_path="configs", 
    config_name="config",
    version_base="1.2",
)

def main(dict_config: DictConfig) -> dict:
    print('Loading config...')
    print(OmegaConf.to_yaml(dict_config))
    return dict_config

if __name__ == "__main__":
    main()
    
    