#!/usr/bin/env python
"""
Debug the actual PyTorch storage API
"""

import torch

def debug_storage_api():
    """Check what storage methods actually exist"""
    
    print("Debugging PyTorch Storage API")
    print("=" * 40)
    
    cpu_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    mps_tensor = cpu_tensor.to("mps")
    
    print(f"CPU tensor: {cpu_tensor}")
    print(f"MPS tensor: {mps_tensor}")
    
    # Check what methods are available on storage
    print(f"\nCPU storage type: {type(cpu_tensor.storage())}")
    print(f"MPS storage type: {type(mps_tensor.storage())}")
    
    print(f"\nCPU storage methods:")
    cpu_storage_methods = [m for m in dir(cpu_tensor.storage()) if not m.startswith('_')]
    print(f"  {cpu_storage_methods}")
    
    print(f"\nMPS storage methods:")  
    mps_storage_methods = [m for m in dir(mps_tensor.storage()) if not m.startswith('_')]
    print(f"  {mps_storage_methods}")
    
    # Try untyped_storage as suggested
    print(f"\nTrying untyped_storage():")
    try:
        cpu_untyped = cpu_tensor.untyped_storage()
        mps_untyped = mps_tensor.untyped_storage()
        print(f"CPU untyped storage: {type(cpu_untyped)}")
        print(f"MPS untyped storage: {type(mps_untyped)}")
        
        print(f"\nUntyped storage methods:")
        untyped_methods = [m for m in dir(mps_untyped) if not m.startswith('_')]
        print(f"  {untyped_methods}")
        
        # Check if data_ptr exists on untyped storage
        if hasattr(mps_untyped, 'data_ptr'):
            mps_data_ptr = mps_untyped.data_ptr()
            print(f"\nMPS untyped storage data_ptr: 0x{mps_data_ptr:x}")
        else:
            print(f"\nNo data_ptr on untyped storage")
            
    except Exception as e:
        print(f"Error with untyped_storage: {e}")
    
    # What about tensor.data_ptr() directly?
    print(f"\nDirect tensor data_ptr:")
    print(f"CPU tensor data_ptr: 0x{cpu_tensor.data_ptr():x}")
    print(f"MPS tensor data_ptr: 0x{mps_tensor.data_ptr():x}")
    
    # Check if the pointers are different
    print(f"\nAre data pointers the same? {cpu_tensor.data_ptr() == mps_tensor.data_ptr()}")

if __name__ == "__main__":
    debug_storage_api()