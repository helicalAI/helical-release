from typing import Optional
from pathlib import Path
from helical.constants.paths import CACHE_DIR_HELICAL
from typing import Literal

class HelixRConfig:
    """HelixRConfig class to store the configuration of the HelixR model.
    
    Parameters
    ----------
    model_name : Literal["helixR-8L"], optional, default = "helixR-8L"
        The name of the model.
    batch_size : int, optional, default = 24
        The batch size
    device : Literal["cpu", "cuda"], optional, default = "cpu"
        The device to use. Either use "cuda" or "cpu".
    nproc: int, optional, default = 1
        Number of processes to use for data processing.
    """
    def __init__(self, 
            model_name: Literal["helixR-8L"] = "helixR-8L",
            batch_size: int = 24,
            device: Literal["cpu", "cuda"] = "cpu",
            nproc: int = 1):

        self.list_of_files_to_download = [
            f"helixr/{model_name}/config.json",
            f"helixr/{model_name}/training_args.bin",
            f"helixr/{model_name}/model.safetensors",
            f"helixr/{model_name}/generation_config.json",
        ]

        self.config = {
            'model_name': model_name,
            'model_dir': Path(CACHE_DIR_HELICAL, 'helixr/', model_name),
            'batch_size': batch_size,
            'device': device,
            'nproc': nproc
        }
