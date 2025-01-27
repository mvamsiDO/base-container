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

### Notes while Testing PT211-TF215-CUDA12 on A100 on U22 machine
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


### Notes while Testing PT211-TF215 on A100 on U22 machine with CUDA 12.0 and CUDANN-8.8.1
- Need to check if torch will work as it is on 12.1 and we are going back a version! -> works fine 
- is 12 cuda tk working on 12.2 driver on host? -> yes, is it might be backward compatable. (host driver enabling this i pressume)
- TF 2.15 fails to work! [Says its compiled using 12.2](https://www.tensorflow.org/install/source#gpu) ; but in MLiaB it works on 12.1.


### Notes while Testing PT211-TF215 on gradient machine(P6000) with CUDA 12.1.1 and CUDANN-8.9.3
- Pushing `cu1211_530_with_latest` to `mvamsi757/gradient_container:cu1211_530` [hub link](https://hub.docker.com/repository/docker/mvamsi757/gradient_container/tags?page=1&ordering=last_updated)
- So, by this logic, 12.1 cuda should not work on 12.0 driver on host -> Yes, it is not Working!! `nvidia-smi: Failed to initialize NVML: Driver/library version mismatch`
- Check if with 12.1 cuda, TF 2.15 works -> Yes, Works fine!
- Torch works? -> Yes, Works fine!
- Interesting observations: So `nvidia-smi` is not working as expected, but the GPU is accessible and being utilised by PyTorch and TF. 
- Lets assume this is ok behaviour or ASK the engg to upgrade the drivers to 530.
- Start tests on various family machines
- Common issues observerd:
    - `nvidia-smi` : `Driver/library version mismatch` (DLVM)
    - `TF2.15.0` : `Loaded runtime CuDNN library: 8.8.1 but source was compiled with: 8.9.4.` (CUDNN-MisMatch)

| GPU/Family       | Nvidia-smi?      | PyTorch?     | Tensorflow?    | CudaNN test? | 
| -------------    | -------------    | -------------| -------------  | -------------|
| P6000            | No, DLVM         | Yes          | Yes            | Yes          |
| RTX4000          | No, DLVM         | Yes          | Yes            | Yes          |
| A6000            | No, DLVM         | Yes          | Yes            | Yes          |
| V100-32          | No, DLVM         | Yes          | Yes            | Yes          |
| A100             | No, DLVM         | Yes          | Yes            | Yes          |

### Tests on PT211-TF215 with CUDA 12.1.1-deb and CUDNN-8.9.3
- Same as before which is what i would have expected!


| GPU/Family       | Nvidia-smi?      | PyTorch?     | Tensorflow?    | CudaNN test? | 
| -------------    | -------------    | -------------| -------------  | -------------|
| P6000            | No, DLVM         | Yes          | Yes            | Yes          |
| RTX4000          | No, DLVM         | Yes          | Yes            | Yes          |
| A6000            | No, DLVM         | Yes          | Yes            | Yes          |
| V100-32          | No, DLVM         | Yes          | Yes            | Yes          |
| A100             | No, DLVM         | Yes          | Yes            | Yes          |


### Notes while Testing PT211-TF215 on gradient machine(P6000) with CUDA 12.0 and CUDANN-8.8.1
- Pushing to `gradient_container:cu120_525` [hub link](https://hub.docker.com/repository/docker/mvamsi757/gradient_container/tags?page=1&ordering=last_updated)
- So,  12.0 cuda should work on 12.0 driver on host -> `nvidia-smi` works fine
- TF works (my guess is no!) -> No it does not, have to downgrade to 2.14
- Torch works? -> Yes!

### Tests on PT211-TF215 with CUDA 12.0-525 and CUDNN-8.9.7
- Docker image here `gradient_container:cu120_525_tf15_cudnn897` [hub link](https://hub.docker.com/repository/docker/mvamsi757/gradient_container/tags?page=1&ordering=last_updated)
- Everything works as expected, need to see if we get approval for non deb based installation.

| GPU/Family       | Nvidia-smi?      | PyTorch?     | Tensorflow?    | CudaNN test? | 
| -------------    | -------------    | -------------| -------------  | -------------|
| P6000            | Yes              | Yes          | Yes            | Yes          |
| RTX4000          | Yes              | Yes          | Yes            | Yes          |
| A6000            | Yes              | Yes          | Yes            | Yes          |
| V100-32          | Yes              | Yes          | Yes            | Yes          |
| A100             | Yes              | Yes          | Yes            | Yes          |
| A100x2           | Yes              | Yes          | Yes            | Yes          |


- Accelerate tests:

| GPU/Family       | Nvidia-smi?      | Accelerate?  | PyTorch-MultiGPU? | 
| -------------    | -------------    | -------------|-------------|
| A100             | Yes              | Yes          | NA          |
| A4000x2          | Yes              | Yes          | Yes         |
| A100x2           | Yes              | Yes          | Yes         |


### Random notes:
- There seems to be difference in behavior when Installing cuda toolkit via 
```
 RUN wget  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
        dpkg -i cuda-keyring_1.0-1_all.deb && \
        apt-get update
RUN $APT_INSTALL cuda-12-0 && \  
    rm cuda-keyring_1.0-1_all.deb
```
AND
```
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \
        mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
        wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb && \
        dpkg -i cuda-repo-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb && \
        cp /var/cuda-repo-ubuntu2204-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
        apt-get update
    RUN $APT_INSTALL cuda && \  
        rm cuda-repo-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb
```
- Docker img gets built via 1st approach 
    - Img name on A100_U22 machine: `cu120_deb_tf214` 
    - Has `12.0.1` versions in `/usr/local/cuda/version.json` 
    - Has `12.0.140` version for `cuda_nvml_dev`
    - `nvidia-smi` does not works fine, throws error: `Failed to initialize NVML: Driver/library version mismatch NVML library version: 545.23`
    - My guess is it will not work on Notebook either? - correct! does not work!
- Docker img gets built via 2nd approach 
    - Img name on A100_U22 machine: `cu120_525_tf214` 
    - Has `12.0.0` versions in `/usr/local/cuda/version.json` 
    - Has `12.0.76` version for `cuda_nvml_dev`
    - `nvidia-smi` works fine, even tough it shows 12.2 + 535 as the cuda driver (which is what the A100 has)
    - My guess is it will work fine on Notebook either? - yes, works!
    - `TF` was not able to detect GPU -> Upgraded to TF 2.15.0 and upgraded the CudNN version to 8.9.7 ->  everything seems to work.

### Compressions Experiement Notes:
- Understood that the docker image is almost double beacuse of the way we are installing Cuda Toolkit. 
- So instead of local deb, tried to install via network deb [ref link](https://developer.nvidia.com/cuda-12-0-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network)
- As it was earlier, the package manager was not picking the rite version of additional libraries; for the ones we mention it picks 12.0.0, but for all the additional stuff it defaults to 12.0.1, and this causes the `DLVM` version mis-match error on `nvidia-smi` ; Check [Dockerfile](cu120_deb_pt211-tf214-accl024-py311/Dockerfile)
-  Finally to reduce the compressed docker image from 15GB
    - Concatinated the `wget`, `apt-get install` and `rm deb` in one Docker `RUN` command -> reduced to compressed 12GB
    - Also, by going to a non-deb based (.run based) installer of `CUDA Toolkit` [ref link](https://developer.nvidia.com/cuda-12-0-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local) -> reduce the compressed size to 9.2GB; Check [Dockerfile](cu120_local_pt211-tf215-accl024-py311/Dockerfile)

- Basic tests on both reduced versions on `A100x2` done and all tests pass

| Docker tag       | Nvidia-smi?      | PyTorch?     | Tensorflow?    | CudaNN test? | 
| -------------    | -------------    | -------------| -------------  | -------------|
| cu120_525_tf15_cudnn897_concat           | Yes              | Yes          | Yes            | Yes          |
| cu120_525_tf15_cudnn897_lconcat          | Yes              | Yes          | Yes            | Yes          |

- Accelerate tests:

| Docker tag       | Nvidia-smi?      | Accelerate?  | PyTorch-MultiGPU? | 
| -------------    | -------------    | -------------|-------------|
| cu120_525_tf15_cudnn897_concat             | Yes              | Yes          | Yes          |
| cu120_525_tf15_cudnn897_lconcat          | Yes              | Yes          | Yes         |


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
- [Support matrix for CudNN and CUDA](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html)