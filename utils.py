import os


def configure_gpu_device(config):
    """
    Set cuda visible deviceds and re-index config.devices
    This is required for device_map="auto" model loading,
    because cuda tries to get all available GPUs available,
    so CUDA_VISIBLE_DEVICES should be set to specified devices.
    After setting CUDA_VISIBLE_DEVICES, torch device index will be reset to start from 0.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in config.devices)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    config.devices = [i for i in range(len(config.devices))]