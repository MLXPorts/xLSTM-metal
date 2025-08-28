#!/usr/bin/env python
"""
Debug what's actually in MPS tensor memory
"""

import torch
import ctypes

def debug_mps_tensor_memory():
    """Look at the actual memory content of MPS tensors"""
    
    print("Debugging MPS Tensor Memory Content")
    print("=" * 50)
    
    # Create simple test data
    cpu_data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
    mps_data = cpu_data.to("mps")
    
    print(f"CPU tensor: {cpu_data}")
    print(f"MPS tensor: {mps_data}")
    print(f"Are they equal? {torch.equal(cpu_data, mps_data.cpu())}")
    
    # Check memory addresses
    cpu_ptr = cpu_data.data_ptr()
    mps_ptr = mps_data.data_ptr() 
    
    print(f"\nMemory addresses:")
    print(f"CPU data_ptr: 0x{cpu_ptr:x}")
    print(f"MPS data_ptr: 0x{mps_ptr:x}")
    
    # Try to read MPS memory directly
    print(f"\nTrying to read MPS memory directly...")
    try:
        # This is what our Metal kernel tries to do
        mps_storage_ptr = mps_data.storage().data()
        print(f"MPS storage data(): {mps_storage_ptr}")
        
        # Check if we can read it as floats
        if mps_storage_ptr:
            float_array = ctypes.cast(mps_storage_ptr, ctypes.POINTER(ctypes.c_float))
            values = [float_array[i] for i in range(5)]
            print(f"Raw memory content: {values}")
        else:
            print("MPS storage data() returned None!")
            
    except Exception as e:
        print(f"Error reading MPS memory: {e}")
    
    # The real test: What does our simple equation produce?
    print(f"\nSimple equation test:")
    a = 15.0
    expected = a * torch.tanh(cpu_data / a)
    mps_result = a * torch.tanh(mps_data / a)
    
    print(f"CPU result: {expected}")
    print(f"MPS result: {mps_result}")
    print(f"MPS result on CPU: {mps_result.cpu()}")

if __name__ == "__main__":
    debug_mps_tensor_memory()