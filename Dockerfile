FROM rayproject/ray-ml:latest-gpu

RUN sudo apt-get update -q \
    && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    patchelf \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    xpra \
    xserver-xorg-dev \
    && sudo apt-get clean \
    && sudo rm -rf /var/lib/apt/lists/*

## MuJoCo
RUN mkdir ~/.mujoco && cd ~/.mujoco \
    && wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz \
    && tar -xf mujoco210-linux-x86_64.tar.gz \
    && rm mujoco210-linux-x86_64.tar.gz 

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ray/.mujoco/mujoco210/bin

# robogym
RUN git clone https://github.com/rodrigodelazcano/robogym.git \
    && cd robogym \
    && pip install -e .

# mujoco_py
RUN git clone https://github.com/openai/mujoco-py.git \
    && cd mujoco-py \
    && pip install -e. --no-cache \
    && python -c "import mujoco_py"


