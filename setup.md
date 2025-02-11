# isaac-sim
## Workstation Installation
### Workstation Setup
1. Install Visual Studio Code
    1. Download [Visual Studio Code](https://code.visualstudio.com/download)
    2. Install
       ```
        sudo apt install ./filename
       ```
2. Install Hub Workstation Cache
    1. Download [Hub Workstation Cache](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/omniverse/resources/hub_workstation_cache)
    2. Install with `./scripts/install.[bat/sh]`
3. Install Isaac Sim 4.5.0
    1. Download [Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/download.html#isaac-sim-latest-release)
    2. Unzip the package
       ```
        mkdir ~/isaacsim
        cd ~/Downloads
        unzip "isaac-sim-standalone@4.5.0-rc.36+release.19112.f59b3005.gl.linux-x86_64.release.zip" -d ~/isaacsim
        cd ~/isaacsim
        ./post_install.sh
        ./isaac-sim.selector.sh
       ```
    3. Run the Isaac Sim app with `./isaac-sim.sh`
    4. Click START to run the Isaac Sim main app
### Container Installation
1. Container Setup
    1. Install NVIDIA Driver
       ```
        sudo apt-get update
        sudo apt install build-essential -y
        wget https://us.download.nvidia.com/XFree86/Linux-x86_64/535.129.03/NVIDIA-Linux-x86_64-535.129.03.run
        chmod +x NVIDIA-Linux-x86_64-535.129.03.run
        sudo ./NVIDIA-Linux-x86_64-535.129.03.run
       ```
       But I passed this step.
    2. Install Docker
       ```
        # Docker installation using the convenience script
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh

        # Post-install steps for Docker
        sudo groupadd docker
        sudo usermod -aG docker $USER
        newgrp docker

        # Verify Docker
        docker run hello-world
       ```
    3. Install NVIDIA Container Toolkit
       ```
        # Configure the repository
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
            && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
            && \
            sudo apt-get update

        # Install the NVIDIA Container Toolkit packages
        sudo apt-get install -y nvidia-container-toolkit
        sudo systemctl restart docker

        # Configure the container runtime
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker

        # Verify NVIDIA Container Toolkit
        docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
       ```
### Container Deployment
1. Run `nvidia-smi`
2. Pull the [Isaac Sim Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim)
   ```
    docker pull nvcr.io/nvidia/isaac-sim:4.5.0
   ```
3. Run the Isaac Sim container with an interactive Bash session
   ```
    docker run --name isaac-sim --entrypoint bash -it --runtime=nvidia --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
    -e "PRIVACY_CONSENT=Y" \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    nvcr.io/nvidia/isaac-sim:4.5.0
   ```
4. Start Isaac Sim with native livestream mode. Before running a livestream client, load the Isaac Sim app.
   ```
    ./runheadless.sh -v
   ```
5. Download and install the [Isaac Sim WebRTC Streaming Client](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/download.html#isaac-sim-latest-release) and run
6. Enter the IP address of the machine or instance running the Isaac Sim container and click on the Connect button to begin live streaming.

## Python Environment Installation
### Install Isaac Sim using PIP
1. Create and activate the virtual environment
   ```
    python3.10 -m venv env_isaacsim
    source env_isaacsim/bin/activate
   ```
2. Install Isaac Sim-Python packages
   ```
    pip install isaacsim[all]==4.5.0 --extra-index-url https://pypi.nvidia.com
   ```
3. Install Isaac Sim-Python packages cached extension dependencies
   ```
    pip install isaacsim[extscache]==4.5.0 --extra-index-url https://pypi.nvidia.com
   ```
4. The Isaac Sim package provides a `.vscode` workspace -> You can open the workspace with VS Code
