#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_main_imports():
    """Test that the main function can be imported and basic components work."""
    try:
        from src.main import main, instantiate_algorithms, instantiate_model
        print("✅ Main function imports successful")
        
        from src.algorithms.pca import PCAModule
        print("✅ PCAModule import successful")
        
        from src.data.dummy import DummyDataModule
        print("✅ DummyDataModule import successful")
        
        from omegaconf import OmegaConf
        print("✅ OmegaConf import successful")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_algorithm_instantiation():
    """Test algorithm instantiation."""
    try:
        from omegaconf import OmegaConf
        from src.main import instantiate_algorithms
        from src.data.dummy import DummyDataModule
        
        # Create a simple datamodule
        datamodule = DummyDataModule(batch_size=32, num_samples=100, input_dim=10)
        datamodule.setup()
        
        # Test algorithm instantiation
        cfg = OmegaConf.create({
            "algorithms": [
                {"_target_": "src.algorithms.pca.PCAModule", "n_components": 2}
            ]
        })
        
        algorithms = instantiate_algorithms(cfg, datamodule)
        print(f"✅ Algorithm instantiation successful: {len(algorithms)} algorithms")
        print(f"   Algorithm type: {type(algorithms[0])}")
        
        return True
    except Exception as e:
        print(f"❌ Algorithm instantiation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_module():
    """Test data module functionality."""
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
        
        # Test get_labels
        labels = datamodule.test_dataset.get_labels()
        print(f"   Test dataset labels shape: {labels.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Data module error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing refactored main function components...")
    
    success = True
    success &= test_main_imports()
    success &= test_algorithm_instantiation()
    success &= test_data_module()
    
    if success:
        print("\n✅ All tests passed!")
        print("The refactored main function should work correctly.")
    else:
        print("\n❌ Some tests failed!")
    
    sys.exit(0 if success else 1) 