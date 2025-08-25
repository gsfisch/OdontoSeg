import torch

print(torch.version.cuda)

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the current device
    device = torch.device("cuda")

    # Print the device properties
    print(f"Using device: {device}")
    print(torch.cuda.get_device_name(device))
    print(f"CUDA version: {torch.version.cuda}")

    # Create a tensor on the GPU
    x = torch.randn(3, 3).to(device)

    # Perform some computations on the tensor
    y = x * 2 + 1

    # Move the tensor back to the CPU and print the result
    print(y.cpu())
else:
    print("CUDA is not available on this device.")

