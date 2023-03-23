# pytorch image base
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel as base
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES},display \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# environment variables
ARG USERNAME=xxx
ENV UDATADIR=/data \
    UPRJDIR=/code \
    UOUTDIR=/output \
    UDEVICEID=docker
ENV FIFTYONE_DATASET_ZOO_DIR=${UDATADIR}/fiftyone \
    FIFTYONE_DEFAULT_DATASET_DIR=${UDATADIR}/fiftyone \
    FIFTYONE_MODEL_ZOO_DIR=${UDATADIR}/fiftyone/__models__ \
    FIFTYONE_DATABASE_DIR=${UDATADIR}/fiftyone/database/mongo


# apt
RUN apt-get update && apt-get install -y \
    curl \
    git \
    sudo \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    htop \
    vim \
    && rm -rf /var/lib/apt/lists/*

# python packages
RUN conda install -y --name base -c conda-forge \
    tensorboard \
    pandas \
    && rm -rf /var/lib/apt/lists/*

RUN pip install \
    opencv-python \
    pytorch-lightning \
    # hydra-joblib-launcher \
    hydra-core \
    hydra-colorlog \
    hydra-optuna-sweeper \
    torchmetrics \
    pyrootutils \
    pre-commit \
    pytest \
    sh \
    omegaconf \
    rich \
    fiftyone \
    jupyter \
    wandb \
    grad-cam \
    tensorboardx \
    ipdb
    # && rm -rf /var/lib/apt/lists/*

RUN pip install \
    hydra-joblib-launcher 

# Non-root user creation and enter
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && mkdir -p /home/$USERNAME/.vscode-server /home/$USERNAME/.vscode-server-insiders \
    && chown ${USER_UID}:${USER_GID} /home/$USERNAME/.vscode-server* \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && usermod -a -G audio,video $USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
USER $USERNAME
ENV HOME /home/$USERNAME
WORKDIR $HOME

# CMD tensorboard --bind_all \
# 	--logdir=${UOUTDIR}/pytorch_output \
# 	& \
CMD	\
    fiftyone app launch --remote \
	--port 5151 \
	-A 0.0.0.0 \
	& \
    sleep infinity