### Notes while Testing Current Version on a Core Machine
- `sudo docker build -t <img_name> .` works as expected
- To Run with GPUs `sudo docker run -d --gpus all  -p 8888:8888  <img_name>` 
- To connect to from local machine to the Docker image running in a remote server:
    ```bash
    ssh -N -f -L localhost:8888:0.0.0.0:8888 paperspace@<remote_server_ip>

    #open browser with localhost:8888
    ```
- Use the `sudo docker logs <image_id>` to find the `token` and enter the same; `jupyter lab` should spin up
- Clone [this repo](https://github.com/mvamsiDO/runtime-notebooks.git) and try running the notebooks under `runtime-notebooks/nbs/`
- Most of the Notebooks work as expected! 
- Issues faced:
    - `misc.ipynb` does not work properly, `import flax` fails -> (can ignore)
    - `cudnn_samples_v8/mnistCUDNN` fails at `make` with the following error: `test.c:1:10: fatal error: FreeImage.h: No such file or directory` -> (can ignore)
    - `accelerate` is not installed, only found it in `ml-in-a-box/ubuntu-22` -> (should get fixed with new versions)

### Notes while Testing PT211-TF215-CUDA12 on A10 on U22 machine
- Spun up a `ubuntu-22` base OS, `docker`, `nvidia-smi` not installed
- Installing the basic pkgs from 1st cmd in MLiab
- `sudo apt install docker.io` to install Docker
- `sudo apt install nvidia-driver-535` ; does not work immediatly, Need to figure out 
    - if a reboot helps -> Sure does!!!
    - or search for approp driver -> Not needed.
- `sudo apt install nvidia-cuda-toolkit` 
- Had to install nvidia-docker
    ```bash
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

    # Install the NVIDIA docker toolkit
    sudo apt-get update
    sudo apt-get install -y nvidia-docker2

    sudo systemctl restart docker
    ```
- Need to see if this [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) is better than `nvidia-docker2`

- Still facing some `cuda toolkit` issues, Running into this when  `accelerate launch --config_file accelerate_condigs/ds-stg2.yaml train_causal_bnb.py` (docker img: same_a100_u20_img)
    ```
    The following directories listed in your path were found to be non-existent: {PosixPath('/tmp/torchelastic_dzgu5pzy/none_b_8v4_ou/attempt_0/0/error.json')}
    CUDA_SETUP: WARNING! libcudart.so not found in any environmental path. Searching in backup paths...
    The following directories listed in your path were found to be non-existent: {PosixPath('/usr/local/cuda/lib64')}
    DEBUG: Possible options found for libcudart.so: set()
    CUDA SETUP: PyTorch settings found: CUDA_VERSION=121, Highest Compute Capability: 8.0.
    CUDA SETUP: To manually override the PyTorch CUDA version please see:https://github.com/TimDettmers/bitsandbytes/blob/main/how_to_use_nonpytorch_cuda.md
    CUDA SETUP: Loading binary /usr/local/lib/python3.11/dist-packages/bitsandbytes/libbitsandbytes_cuda121.so...
    libcusparse.so.12: cannot open shared object file: No such file or directory
    CUDA SETUP: Problem: The main issue seems to be that the main CUDA runtime library was not detected.
    CUDA SETUP: Solution 1: To solve the issue the libcudart.so location needs to be added to the LD_LIBRARY_PATH variable
    CUDA SETUP: Solution 1a): Find the cuda runtime library via: find / -name libcudart.so 2>/dev/null
    CUDA SETUP: Solution 1b): Once the library is found add it to the LD_LIBRARY_PATH: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:FOUND_PATH_FROM_1a
    CUDA SETUP: Solution 1c): For a permanent solution add the export from 1b into your .bashrc file, located at ~/.bashrc
    CUDA SETUP: Solution 2: If no library was found in step 1a) you need to install CUDA.
    CUDA SETUP: Solution 2a): Download CUDA install script: wget https://github.com/TimDettmers/bitsandbytes/blob/main/cuda_install.sh
    CUDA SETUP: Solution 2b): Install desired CUDA version to desired location. The syntax is bash cuda_install.sh CUDA_VERSION PATH_TO_INSTALL_INTO.
    CUDA SETUP: Solution 2b): For example, "bash cuda_install.sh 113 ~/local/" will download CUDA 11.3 and install into the folder ~/local
    ```
    
    - Trying [this soln here](https://stackoverflow.com/a/62791665) -> Does not work, ignore; reverted!
    
    - Making an env variable that can share all the libs with docker container. -> Does not work!
        ```
        # adding this to docker file
        ENV LD_LIBRARY_PATH /usr/lib/from_host/

        #and running like so:
        docker run -v /usr/lib/x86_64-linux-gnu/:/usr/lib/from_host/ myimage
        ```
    
    - This seems to work, integrating into Docker. 
        ```
            #Download CUDA install script: 
            wget https://github.com/TimDettmers/bitsandbytes/blob/main/install_cuda.sh
            export LD_LIBRARY_PATH
            export PATH

            # Something like this:
            sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub && \
            sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" && \
            sudo apt-get update && \
            sudo apt-get install cuda=12.2.0-1 

            # OR
            wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
            sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
            wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb
            sudo dpkg -i cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb
            sudo cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
            sudo apt-get update
            sudo apt-get -y install cuda
        ```
    - This CUDA installation above works and Torch, accelerate seem to work fine, TF still causing issues when we remove pip based cuda stuff
        - Need to verify if TF is able to access GPU on an ML-In-Box machine? -> Yes!
        - Got it to work with `libcudnn8` install; incorprating it into Docker -> Done!

- **Learning:** We do not need `pip install tensorflow[and-cuda]` and can use `pip install tensorflow=2.15.0` directly, as long as the `cuda-toolkit` and `libcudnn` are installed and path is specified properly.

### Some Useful commands:
```bash
# build docker image
sudo docker build -t <img_name> .

# run with gpus
sudo docker run -d --gpus all  -p 8888:8888  <img_name>

#search for PIDs using a particular port
lsof -ti:8888

#see what exactly is running in the PID
ps -ef | grep <PID>   

#check GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

#check accelerate base setup
python -m bitsandbytes

#nv-smi
nvidia-smi

```

### Some Useful resources:
- GPU - CUDA compatibility : https://docs.nvidia.com/deploy/cuda-compatibility/
- [Useful Link](https://gitlab.com/nvidia/container-images/cuda/blob/master/dist/12.3.1/ubuntu2204/base/Dockerfile) found to get `apt get install` to work with `cuda, cuda toolkit etc`
- [Fix to the CUDNN Sample run bug](https://forums.developer.nvidia.com/t/freeimage-is-not-set-up-correctly-please-ensure-freeimae-is-set-up-correctly/66950/3)