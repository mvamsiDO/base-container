# Needs to Get Updated!
# Gradient Base Image

This repo houses a Dockerfile used to create an image used for Gradient runtimes. This image should be used for all Gradient runtimes, either solely or as the base layer for other images.
 
The goal of this image is to provide common packages for an advanced data science user that allows them to utilize the GPU hardware on Gradient. This image targets popular Machine Learning frameworks and includes packages commonly used for Computer Vision and Natural Language Processing tasks as well as general Data Science work.


## Software included

| Category         | Software         | Version                | Install Method | Why / Notes |
| -------------    | -------------    | -------------          | -------------  | ------------- |
| GPU              | NVidia Driver    | 510.73.05              | pre-installed  | Enable Nvidia GPUs |
|                  | CUDA             | 11.6.2                 | Apt            | Nvidia A100 GPUs require CUDA 11+ to work, so 10.x is not suitable |
|                  | CUDA toolkit     | 11.6.2                 | Apt            | Needed for `nvcc` command for cuDNN |
|                  | cuDNN            | 8.4.1.*-1+cuda11.6     | Apt            | Nvidia GPU deep learning library |
| Python           | Python           | 3.9.15                 | Apt            | Most widely used programming language for data science |
|                  | pip3             | 22.2.2                 | Apt            | Enable easy installation of 1000s of other data science, etc., packages. |
|                  | NumPy            | 1.23.4                 | pip3           | Handle arrays, matrices, etc., in Python |
|                  | SciPy            | 1.9.2                  | pip3           | Fundamental algorithms for scientific computing in Python |
|                  | Pandas           | 1.5.0                  | pip3           | De facto standard for data science data exploration/preparation in Python |
|                  | Cloudpickle      | 2.2.0                  | pip3           | Makes it possible to serialize Python constructs not supported by the default pickle module |
|                  | Matplotlib       | 3.6.1                  | pip3           | Widely used plotting library in Python for data science, e.g., scikit-learn plotting requires it |
|                  | Ipython          | 8.5.0                  | pip3           | Provides a rich architecture for interactive computing |
|                  | IPykernel        | 6.16.0                 | pip3           | Provides the IPython kernel for Jupyter. |
|                  | IPywidgets       | 8.0.2                  | pip3           | Interactive HTML widgets for Jupyter notebooks and the IPython kernel | 
|                  | Cython           | 0.29.32                | pip3           | Enables writing C extensions for Python |  
|                  | tqdm             | 4.64.1                 | pip3           | Fast, extensible progress meter |  
|                  | gdown            | 4.5.1                  | pip3           | Google drive direct download of big files |  
|                  | Pillow           | 9.2.0                  | pip3           | Python imaging library |  
|                  | seaborn          | 0.12.0                 | pip3           | Python visualization library based on matplotlib |
|                  | SQLAlchemy       | 1.4.41                 | pip3           | Python SQL toolkit and Object Relational Mapper that gives application developers the full power and flexibility of SQL |  
|                  | spaCy            | 3.4.1                  | pip3           | library for advanced Natural Language Processing in Python and Cython |  
|                  | nltk             | 3.7                    | pip3           | Natural Language Toolkit (NLTK) is a Python package for natural language processing |  
|                  | boto3            | 1.24.90                | pip3           | Amazon Web Services (AWS) Software Development Kit (SDK) for Python |  
|                  | tabulate         | 0.9.0                 | pip3           | Pretty-print tabular data in Python |  
|                  | future           | 0.18.2                 | pip3           | The missing compatibility layer between Python 2 and Python 3 |  
|                  | gradient         | 2.0.6                  | pip3           | CLI and Python SDK for Paperspace Core and Gradient |  
|                  | jsonify          | 0.5                    | pip3           | Provides the ability to take a .csv file as input and outputs a file with the same data in .json format |  
|                  | opencv-python    | 4.6.0.66               | pip3           | Includes several hundreds of computer vision algorithms |   
|                  | JupyterLab       | 3.4.6                  | pip3           | De facto standard for data science using Jupyter notebooks |
|                  | wandb            | 0.13.4                 | pip3           | CLI and library to interact with the Weights & Biases API (model tracking) |
| Machine Learning | Scikit-learn     | 1.1.2                  | pip3           | Widely used ML library for data science, generally for smaller data or models |
|                  | Scikit-image     | 0.19.3                 | pip3           | Collection of algorithms for image processing |
|                  | TensorFlow       | 2.9.2                  | pip3           | Most widely used deep learning library, alongside PyTorch |
|                  | torch            | 1.12.1                 | pip3           | Most widely used deep learning library, alongside TensorFlow |
|                  | torchvision      | 0.13.1                 | pip3           | Most widely used deep learning library, alongside TensorFlow |
|                  | torchaudio       | 0.12.1                 | pip3           | Most widely used deep learning library, alongside TensorFlow |
|                  | Jax              | 0.3.23                 | pip3           | Popular deep learning library brought to you by Google |
|                  | Transformers     | 4.21.3                 | pip3           | Popular deep learning library for NLP brought to you by HuggingFace |
|                  | Datasets         | 2.4.0                  | pip3           | A supporting library for NLP use cases and the Transformers library brought to you by HuggingFace |
|                  | XGBoost          | 1.6.2                  | pip3           | An optimized distributed gradient boosting library |
|                  | Sentence Transformers | 2.2.2             | pip3           | A ML framework for sentence, paragraph and image embeddings |

### Licenses

| Software              | License                | Source |
| ---------------       | -------------          | ------------- |
| CUDA 	                | NVidia EULA		  	 | https://docs.nvidia.com/cuda/eula/index.html |
| cuDNN                 | NVidia EULA            | https://docs.nvidia.com/deeplearning/cudnn/sla/index.html |
| JupyterLab            | New BSD      	         | https://github.com/jupyterlab/jupyterlab/blob/master/LICENSE |
| Matplotlib            | PSF-based      		 | https://matplotlib.org/stable/users/license.html |
| Numpy      	        | New BSD                | https://numpy.org/doc/stable/license.html |
| NVidia Docker         | Apache 2.0             | https://github.com/NVIDIA/nvidia-docker/blob/master/LICENSE |
| NVidia Driver         | NVidia EULA            | https://www.nvidia.com/en-us/drivers/nvidia-license/ |
| Pandas                | New BSD                | https://github.com/pandas-dev/pandas/blob/master/LICENSE |
| Pip3                  | MIT                    | https://github.com/pypa/pip/blob/main/LICENSE.txt |
| Python                | PSF                    | https://en.wikipedia.org/wiki/Python_(programming_language) |
| Scikit-learn          | New BSD                | https://github.com/scikit-learn/scikit-learn/blob/main/COPYING |
| Scikit-image          | New BSD                | https://github.com/scikit-image/scikit-image/blob/main/LICENSE.txt |
| TensorFlow            | Apache 2.0             | https://github.com/tensorflow/tensorflow/blob/master/LICENSE |
| PyTorch               | New BSD                | https://github.com/pytorch/pytorch/blob/master/LICENSE |
| Jax                   | Apache 2.0             | https://github.com/google/jax/blob/main/LICENSE |
| Transformers          | Apache 2.0             | https://github.com/huggingface/transformers/blob/main/LICENSE |
| Datasets              | Apache 2.0             | https://github.com/huggingface/datasets/blob/main/LICENSE |
| XGBoost               | Apache 2.0             | https://github.com/dmlc/xgboost/blob/master/LICENSE |
| Sentence Transformers | Apache 2.0             | https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE |
| SciPy                 | New BSD                | https://github.com/scipy/scipy/blob/main/LICENSE.txt |
| Cloudpickle           | New BSD                | https://github.com/cloudpipe/cloudpickle/blob/master/LICENSE |
| Ipython               | New BSD                | https://github.com/ipython/ipython/blob/main/LICENSE |
| IPykernel             | New BSD                | https://github.com/ipython/ipykernel/blob/main/COPYING.md |
| IPywidgets            | New BSD                | https://github.com/jupyter-widgets/ipywidgets/blob/master/LICENSE |
| Cython                | Apache 2.0             | https://github.com/cython/cython/blob/master/LICENSE.txt |
| tqdm                  | MIT                    | https://github.com/tqdm/tqdm/blob/master/LICENCE |
| gdown                 | MIT                    | https://github.com/wkentaro/gdown/blob/main/LICENSE |
| Pillow                | HPND                   | https://github.com/python-pillow/Pillow/blob/main/LICENSE |
| seaborn               | New BSD                | https://github.com/mwaskom/seaborn/blob/master/LICENSE.md |
| SQLAlchemy            | MIT                    | https://github.com/sqlalchemy/sqlalchemy/blob/main/LICENSE |
| spaCy                 | MIT                    | https://github.com/explosion/spaCy/blob/master/LICENSE |
| nltk                  | Apache 2.0             | https://github.com/nltk/nltk/blob/develop/LICENSE.txt |
| boto3                 | Apache 2.0             | https://github.com/boto/boto3/blob/develop/LICENSE |
| tabulate              | MIT                    | https://github.com/astanin/python-tabulate/blob/master/LICENSE |
| future                | MIT                    | https://github.com/PythonCharmers/python-future/blob/master/LICENSE.txt |
| gradient              | ISC                    | https://github.com/Paperspace/gradient-cli/blob/master/LICENSE.txt |
| jsonify               | MIT                    | https://pypi.org/project/jsonify/0.5/#data |
| opencv-python         | MIT                    | https://github.com/opencv/opencv-python/blob/4.x/LICENSE.txt |
| wandb                 | MIT                    | https://github.com/wandb/wandb/blob/master/LICENSE |


Information about license types:

Apache 2.0: https://opensource.org/licenses/Apache-2.0  
MIT: https://opensource.org/licenses/MIT  
New BSD: https://opensource.org/licenses/BSD-3-Clause  
PSF = Python Software Foundation: https://en.wikipedia.org/wiki/Python_Software_Foundation_License
HPND = Historical Permission Notice and Disclaimer: https://opensource.org/licenses/HPND
ISC: https://opensource.org/licenses/ISC

Open source software can be used for commercial purposes: https://opensource.org/docs/osd#fields-of-endeavor.


## Software not included

Other software considered but not included.

The potential data science stack is far larger than any one person will use so we don't attempt to cover everything here.

Some generic categories of software not included:

 - Non-data-science software
 - Commercial software
 - Software not licensed to be used on an available VM template
 - Software only used in particular specialized data science subfields (although we assume our users probably want a GPU)

| Category           | Software | Why Not |
| -------------      | ------------- | ------------- |
| Apache             | Kafka, Parquet | |
| Classifiers        | libsvm | H2O contains SVM and GBM, save on installs |
| Collections        | ELKI, GNU Octave, Weka, Mahout | |
| Connectors         | Academic Torrents | |
| Dashboarding       | panel, dash, voila, streamlit | |
| Databases          | MySQL, Hive, PostgreSQL, Prometheus, Neo4j, MongoDB, Cassandra, Redis | No particular infra to connect to databases |
| Deep Learning      | Caffe, Caffe2, Theano, PaddlePaddle, Chainer, MXNet | PyTorch and TensorFlow are dominant, rest niche |
| Deployment         | Dash, TFServing, R Shiny, Flask | Use Gradient Deployments |
| Distributed        | Horovod, OpenMPI | Use Gradient distributed |
| Feature store      | Feast | |
| Interpretability   | LIME/SHAP, Fairlearn, AI Fairness 360, InterpretML | |
| Languages          | R, SQL, Julia, C++, JavaScript, Python2, Scala | Python is dominant for data science |
| Monitoring         | Grafana | |
| NLP                | GenSim | |
| Notebooks          | Jupyter, Zeppelin | JupyterLab includes Jupyter notebook |
| Orchestrators      | Kubernetes | Use Gradient cluster|
| Partners           | fast.ai | Currently support a specific fast.ai runtime |
| Pipelines          | AirFlow, MLFlow, Intake, Kubeflow | |
| Python libraries   | statsmodels, pymc3, geopandas, Geopy, LIBSVM | Too many to attempt to cover |
| PyTorch extensions | Lightning | |
| R packages         | ggplot, tidyverse | Could add R if customer demand |
| Recommenders       | TFRS, scikit-surprise | |
| Scalable           | Dask, Numba, Spark 1 or 2, Koalas, Hadoop | |
| TensorFlow         | TF 1.15, Recommenders, TensorBoard, TensorRT | Could add TensorFlow 1.x if customer demand. Requires separate tensorflow-gpu for GPU support. |
| Viz                | Bokeh, Plotly, Holoviz (Datashader), Google FACETS, Excalidraw, GraphViz, ggplot2, d3.js | |


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