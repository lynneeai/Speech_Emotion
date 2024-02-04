import os


def configure_gpu_device(devices=None, config=None):
    """
    Set cuda visible deviceds and re-index config.devices
    This is required for device_map="auto" model loading,
    because cuda tries to get all available GPUs available,
    so CUDA_VISIBLE_DEVICES should be set to specified devices.
    After setting CUDA_VISIBLE_DEVICES, torch device index will be reset to start from 0.
    """
    try:
        assert (devices is not None) ^ (config is not None)
    except AssertionError:
        raise AssertionError("Please input either devices or config.")
    
    devices = devices if devices else config.devices
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in devices)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if config:
        config.devices = [i for i in range(len(config.devices))]