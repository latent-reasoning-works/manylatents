#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from omegaconf import OmegaConf
from src.main import main

def test_refactored_main():
    """Test the refactored main function with a simple configuration."""
    
    # Create a simple test configuration
    cfg = OmegaConf.create({
        "algorithms": [
            {"_target_": "src.algorithms.pca.PCAModule", "n_components": 2},
        ],
        "trainer": {
            "max_epochs": 1,
            "gpus": 0
        },
        "callbacks": {"embedding": {}},
        "data": {
            "_target_": "src.data.dummy.DummyDataModule",
            "batch_size": 32,
            "num_samples": 100,
            "input_dim": 10
        },
        "seed": 42,
        "debug": True,
        "project": "test",
        "eval_only": False,
        "cache_dir": "./cache"
    })
    
    print("Testing refactored main function...")
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    try:
        result = main(cfg)
        print("✅ Main function executed successfully!")
        print("Result type:", type(result))
        print("Result keys:", list(result.keys()) if isinstance(result, dict) else "Not a dict")
        
        if isinstance(result, dict):
            print("✅ Result is a dictionary as expected")
            if "embeddings" in result:
                print("✅ Contains embeddings")
                print(f"   Embeddings shape: {result['embeddings'].shape}")
            if "metadata" in result:
                print("✅ Contains metadata")
            if "scores" in result:
                print("✅ Contains scores")
                
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_refactored_main()
    sys.exit(0 if success else 1) 