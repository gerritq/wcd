import torch

def check_torch_setup():
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {n_gpus}")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device capability: {torch.cuda.get_device_capability()}")
    else:
        print("No GPU detected — using CPU.")

if __name__ == "__main__":
    check_torch_setup()