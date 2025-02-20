from dataclasses import dataclass, field
from typing import Any, Optional
import random

@dataclass
class Config:
    algorithm: Any
    datamodule: Optional[Any] = None
    trainer: dict = field(default_factory=dict)
    log_level: str = "info"
    seed: int = field(default_factory=lambda: random.randint(0, int(1e5)))
    debug: bool = False
    verbose: bool = False
    cache_dir: Optional[str] = "outputs/cache"