import sys
import subprocess
import os

def collect_env():
    """[collection env version and basic cuda env]
    """
    env_info = dict()
    env_info['sys.platform'] = sys.platform
    env_info['Python'] = sys.version.replace('\n', '')
    # pytroch and cuda setting
    try:
        import torch
        env_info['torch'] = torch.__version__
        try:
            cuda_available = torch.cuda.is_available()
            if not cuda_available:
                env_info['CUDA'] = 'Not Available'
            else:
                from torch.utils.cpp_extension import CUDA_HOME
                env_info['CUDA_HOME'] = CUDA_HOME
                if CUDA_HOME is not None and os.path.isdir(CUDA_HOME):
                    try:
                        nvcc = os.path.join(CUDA_HOME, 'bin/nvcc')
                        nvcc = subprocess.check_output(
                            f'"{nvcc}" -V | tail -n1', shell=True)
                        nvcc = nvcc.decode('utf-8').strip()
                    except subprocess.SubprocessError:
                        nvcc = 'Not Available'
                    env_info['NVCC'] = nvcc
                else:
                    env_info['NVCC'] = 'Not Available'          
        except:
            env_info['CUDA'] = 'Not Available'
    except ModuleNotFoundError:
        env_info['torch'] = 'NOT FOUND'
    
    try:
        import torchvision
        env_info['TorchVision'] = torchvision.__version__
    except ModuleNotFoundError:
        pass

    # gcc setting 
    try:
        gcc = subprocess.check_output('gcc --version | head -n1', shell=True)
        gcc = gcc.decode('utf-8').strip()
        env_info['GCC'] = gcc
    except subprocess.CalledProcessError:  # gcc is unavailable
        env_info['GCC'] = 'n/a'

    
    # third part package, 
    # if you want to add more kind of pakage you want, just follow the code below
    try:
        import cv2
        env_info['cv2'] = cv2.__version__
    except ModuleNotFoundError:
        env_info['cv2'] = 'Module Not Found !'
        pass

    return env_info

if __name__ == "__main__":
    import pprint
    pprint.pprint(collect_env())