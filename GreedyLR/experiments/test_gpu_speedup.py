#!/usr/bin/env python3
"""
Test GPU speedup independently with MPS vs CPU
"""

import torch
import torch.nn as nn
import time
import sys
sys.path.append('/Users/subshrey/Projects/greedylr_research')

# Simple test model similar to our experiments
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(20, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# Complex transformer model like our experiments
class TestTransformer(nn.Module):
    def __init__(self, dim=256, depth=4, num_heads=8):
        super().__init__()
        self.embed = nn.Linear(20, dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, 
                nhead=num_heads,
                dim_feedforward=dim*4,
                batch_first=True
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)
        
    def forward(self, x):
        x = self.embed(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        return self.head(x.squeeze(1))

def benchmark_device(device_name):
    print(f"\n🧪 Testing {device_name}...")
    
    if device_name == "mps":
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Test data
    X = torch.randn(500, 20).to(device)
    y = torch.randn(500, 1).to(device)
    
    results = {}
    
    for model_name, model_class in [("Simple", TestModel), ("Transformer", TestTransformer)]:
        print(f"  📊 {model_name} model...")
        
        model = model_class().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        # Warmup
        for _ in range(5):
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Sync for accurate timing
        if device_name == "mps":
            torch.mps.synchronize()
        
        # Benchmark 200 steps (same as experiment)
        start_time = time.time()
        
        for step in range(200):
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Sync for accurate timing
        if device_name == "mps":
            torch.mps.synchronize()
            
        elapsed = time.time() - start_time
        results[model_name] = elapsed
        
        print(f"    ⏱️  200 steps: {elapsed:.2f}s ({elapsed/200:.3f}s per step)")
    
    return results

def main():
    print("🚀 GPU Speedup Test")
    print("=" * 50)
    
    # Test CPU
    cpu_results = benchmark_device("cpu")
    
    # Test MPS if available
    if torch.backends.mps.is_available():
        mps_results = benchmark_device("mps")
        
        print("\n📈 SPEEDUP ANALYSIS:")
        print("=" * 30)
        for model_name in cpu_results:
            cpu_time = cpu_results[model_name]
            mps_time = mps_results[model_name]
            speedup = cpu_time / mps_time
            
            print(f"{model_name:12}: {speedup:.1f}x faster on MPS")
            print(f"             CPU: {cpu_time:.2f}s, MPS: {mps_time:.2f}s")
        
        # Estimate experiment speedup
        print(f"\n🎯 EXPERIMENT IMPACT:")
        avg_speedup = sum(cpu_results[m] / mps_results[m] for m in cpu_results) / len(cpu_results)
        print(f"Average speedup: {avg_speedup:.1f}x")
        
        current_time_per_exp = 15  # seconds (observed)
        mps_time_per_exp = current_time_per_exp / avg_speedup
        print(f"Current: ~{current_time_per_exp}s/experiment")
        print(f"With MPS: ~{mps_time_per_exp:.1f}s/experiment")
        
        total_experiments = 669735
        current_total_hours = (total_experiments * current_time_per_exp) / 3600
        mps_total_hours = (total_experiments * mps_time_per_exp) / 3600
        
        print(f"\nTotal experiment time:")
        print(f"CPU-only: {current_total_hours:.0f} hours ({current_total_hours/24:.1f} days)")
        print(f"With MPS: {mps_total_hours:.0f} hours ({mps_total_hours/24:.1f} days)")
        
    else:
        print("\n❌ MPS not available - CPU only")

if __name__ == "__main__":
    main()