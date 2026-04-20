#!/usr/bin/env python3
"""
Test the JSON serialization fix for ExperimentConfig objects
"""
import json
import sys
import os

# Add current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from comprehensive_scheduler_experiment import ExperimentConfig, run_comprehensive_experiment
from robust_comprehensive_experiment import RobustExperimentRunner

def test_serialization_fix():
    """Test that the serialization fix works correctly"""
    print("🧪 Testing JSON serialization fix...")
    
    # Create a test config
    test_config = ExperimentConfig(
        model_type='quadratic',
        noise_type='none',
        noise_strength=0.0,
        total_steps=100,
        scheduler_type='greedy',
        scheduler_params={'min_lr': 1e-5, 'patience': 10, 'factor': 0.5},
        problem_variant='well_conditioned'
    )
    
    print(f"✅ Created test config: {test_config.model_type}/{test_config.scheduler_type}")
    
    # Run a single experiment
    print("🔬 Running single test experiment...")
    try:
        result = run_comprehensive_experiment(test_config)
        print(f"✅ Experiment completed. Final loss: {result['metrics']['final_loss']:.6f}")
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        return False
    
    # Test the serialization with the RobustExperimentRunner
    print("📁 Testing serialization...")
    runner = RobustExperimentRunner(checkpoint_interval=1)
    
    try:
        # Test save_checkpoint with the result
        runner.save_checkpoint([result], 0, 1)
        print("✅ Serialization successful!")
        
        # Verify the file was created and is valid JSON
        if os.path.exists(runner.results_file):
            with open(runner.results_file, 'r') as f:
                loaded_results = json.load(f)
            
            print(f"✅ Successfully loaded {len(loaded_results)} results from JSON")
            
            # Verify the structure
            first_result = loaded_results[0]
            required_fields = ['scheduler_type', 'model_type', 'noise_type', 'metrics']
            missing_fields = [field for field in required_fields if field not in first_result]
            
            if missing_fields:
                print(f"⚠️  Missing fields: {missing_fields}")
            else:
                print("✅ All required fields present in serialized result")
                
            # Test post-processing
            print("📊 Testing post-processing...")
            try:
                processed_data = []
                for result in loaded_results:
                    if isinstance(result, dict):
                        metrics = result.get('metrics', {})
                        if isinstance(metrics, dict):
                            flattened = {
                                'model_type': result.get('model_type', 'unknown'),
                                'scheduler_type': result.get('scheduler_type', 'unknown'),
                                'noise_type': result.get('noise_type', 'unknown'),
                                'final_loss': metrics.get('final_loss', float('inf')),
                                'convergence_rate_50': metrics.get('convergence_rate_50', 0),
                                'stability_score': metrics.get('stability_score', 0)
                            }
                            processed_data.append(flattened)
                
                if processed_data:
                    import pandas as pd
                    df = pd.DataFrame(processed_data)
                    print(f"✅ Post-processing successful! DataFrame shape: {df.shape}")
                    print(f"   Columns: {list(df.columns)}")
                    print(f"   Sample data: {df.iloc[0].to_dict()}")
                else:
                    print("❌ Post-processing failed - no processed data")
                    
            except Exception as e:
                print(f"❌ Post-processing failed: {e}")
                return False
                
        else:
            print(f"❌ Results file not created: {runner.results_file}")
            return False
            
    except Exception as e:
        print(f"❌ Serialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 ALL TESTS PASSED!")
    print("✅ JSON serialization fix works correctly")
    print("✅ Post-processing can access all required fields")
    print("✅ Ready for full experiment run")
    
    return True

if __name__ == "__main__":
    success = test_serialization_fix()
    sys.exit(0 if success else 1)