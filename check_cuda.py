import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Test actual GPU operation
    try:
        x = torch.randn(10, 10).cuda()
        y = x * 2
        print("✓ GPU test successful - CUDA is working!")
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
else:
    print("CUDA not available")