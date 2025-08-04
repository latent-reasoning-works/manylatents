#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all necessary imports work."""
    try:
        from src.algorithms.pca import PCAModule
        print("✅ PCAModule import successful")
        
        from src.data.dummy import DummyDataModule
        print("✅ DummyDataModule import successful")
        
        from src.main import instantiate_algorithms
        print("✅ instantiate_algorithms import successful")
        
        from omegaconf import OmegaConf
        print("✅ OmegaConf import successful")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_algorithm_instantiation():
    """Test that algorithm instantiation works."""
    try:
        from omegaconf import OmegaConf
        from src.main import instantiate_algorithms
        
        cfg = OmegaConf.create({
            "algorithms": [
                {"_target_": "src.algorithms.pca.PCAModule", "n_components": 2}
            ]
        })
        
        algorithms = instantiate_algorithms(cfg)
        print(f"✅ Algorithm instantiation successful: {len(algorithms)} algorithms")
        print(f"   Algorithm type: {type(algorithms[0])}")
        
        return True
    except Exception as e:
        print(f"❌ Algorithm instantiation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_module():
    """Test that data module works."""
    try:
        from src.data.dummy import DummyDataModule
        
        datamodule = DummyDataModule(batch_size=32, num_samples=100, input_dim=10)
        datamodule.setup()
        
        train_loader = datamodule.train_dataloader()
        test_loader = datamodule.test_dataloader()
        
        print(f"✅ Data module setup successful")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        
        # Test first batch
        first_batch = next(iter(train_loader))
        print(f"   First batch keys: {list(first_batch.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Data module error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing refactored code components...")
    
    success = True
    success &= test_imports()
    success &= test_algorithm_instantiation()
    success &= test_data_module()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    sys.exit(0 if success else 1) 