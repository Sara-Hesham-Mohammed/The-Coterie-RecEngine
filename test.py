import torch

# Check available GPUs
print(f"Available GPUs: {torch.cuda.device_count()}")

# Check current GPU being used (default is 0)
print(f"Current GPU: {torch.cuda.current_device()}")

# Get GPU name
print(f"GPU Name: {torch.cuda.get_device_name(0)}")  # Replace 0 with GPU index