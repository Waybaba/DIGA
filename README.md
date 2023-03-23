# (DIGA)Dynamically Instance-Guided Adaptation: A Backward-free Approach for Test-Time Domain Adaptive Semantic Segmentation

# Environment Setup

Before everything start, please make sure the following environment variables(`UDATADIR, UPRJDIR, UOUTDIR`) are setup.  The following operation would only modifying the above three folders on your devices. 

example:

```python
# e.g. 
export UDATADIR=~/data # dir for dataset
export UPRJDIR=~/code # dir for code
export UOUTDIR=~/output # dir for output such as logs
export WANDB_API_KEY="xxx360492802218be41f76xxxx" # your wandb key
export NUM_WORKERS=0 # number of works used
mkdir -p $UDATADIR $UDATADIR $UOUTDIR # create dir if it does not exist
```

ps. Wandb is a wonderful tool for visualization similar to Tensorboard but offer more functions ([link](https://docs.wandb.ai/quickstart)). 

## Code Setup

1. Copy or clone this project.
2. Organize the files in the following structure.

Code for this project should be put in `$UPRJDIR` with the following structure:

```python
$UPRJDIR
	├── DIGA
			├── ...
			├── Readme.md
			├── ...
```

Then you can use `cd $UPRJDIR/DIGA` to get in the project directory.

## Dataset Setup

1. Init file structure by running `cp -r $UPRJDIR/DIGA/src/utils/advent_list_lib/* $UDATADIR/`
2. download dataset

Please read the following before get start.

The environment variable `$UDATADIR` setted in last step would be used to locate the dataset files when program is running. Each dataset is contained in one folder, the structure in `$UDATADIR` should be like this: 

```python
$UDATADIR
	├── GTA5 # (GTA5 in paper)
	├── synthia # (Synthia in paper)
	├── Cityscapes # (CityScapes in paper)
	├── BDD # (BDD100K in paper)
	├── IDD # (IDD in paper)
	├── NTHU # (CrossCity in paper)
	├── Mapillary # (Mapillary in paper)
```

In most dataset folders, there is one folder called “**advent_list**”, which contains the list for training, validation, testing set [following implementation in [ADVENT](https://github.com/valeoai/ADVENT)]. 

We provide the lists file in this project, the folder is `$UPRJDIR/DIGA/src/utils/advent_list_lib`

Running the follows would set up the folders for you automatically:

```python
# copy DIGA/src/utils/advent_list_lib/* to $UDATADIR/
cp -r $UPRJDIR/DIGA/src/utils/advent_list_lib/* $UDATADIR/
```

Now, the structure should look like this:

```python
$UDATADIR
	├── GTA5 # (GTA5 in paper)
	│   ├── advent_list
	│   ├── ...
	├── synthia # (Synthia in paper)
	│   ├── ...
	│   ├── ...
	├── Cityscapes # (CityScapes in paper)
	│   ├── advent_list
	│   ├── ...
	├── BDD # (BDD100K in paper)
	│   ├── advent_list
	│   ├── ...
	├── IDD # (IDD in paper)
	│   ├── ...
	│   ├── ...
	├── NTHU # (CrossCity in paper)
	│   ├── advent_list
	│   ├── ...
	├── Mapillary # (Mapillary in paper)
	│   ├── advent_list
	│   ├── ...
```

The next step is downloading each dataset. It would cost too much space to offer instructions about downloading and there are many tutorial online. Instead, we offer an illustration as follows about the final dataset structure. You can download the datasets and organize them like follows. Some instructions can be find here ([link1](https://github.com/wasidennis/AdaptSegNet), [link2](https://github.com/valeoai/ADVENT))

```python
$UDATADIR
├── BDD
│   ├── advent_list
│   ├── images
│   ├── labels
├── Cityscapes
│   ├── README
│   ├── advent_list
│   ├── gtFine
│   ├── leftImg8bit
├── GTA5
│   ├── advent_list
│   ├── images
│   ├── labels
├── IDD
│   ├── gtFine
│   ├── iddscripts
│   ├── leftImg8bit
├── Mapillary
│   ├── LICENSE
│   ├── README
│   ├── advent_list
│   ├── config.json
│   ├── small
│   ├── testing
│   ├── training
│   └── validation
├── NTHU
│   ├── Rio
│   ├── Rome
│   ├── Taipei
│   ├── Tokyo
│   └── advent_list
├── synthia
│   ├── Depth
│   ├── GT
│   ├── RGB
```

## Pre-trained model Setup

1. Download pre-trained models
2. Organize the files in the following structure.

Source models are required to be put in specific directory for running. The pre-trained models are available here ([link](https://www.dropbox.com/s/gpzm15ipyt01mis/DA_Seg_models.zip?dl=0)). 

ps. We use consistent models from previous work of this repo [link](https://github.com/wasidennis/AdaptSegNet).

```python
$UDATADIR
├── models # create a new folder called "models" under $UDATADIR 
		├── DA_Seg_models # downloaded from above link
				├── GTA5
						├── GTA5_baseline.pth
				├── SYNTHIA
						├── SYNTHIA_source.pth
				├── ...
├── ...
├── BDD
```

The following cmd would do this automatically:

```python
mkdir -p $UDATADIR/models
cd $UDATADIR/models
wget -O download.zip https://www.dropbox.com/s/gpzm15ipyt01mis/DA_Seg_models.zip?dl=0https://www.dropbox.com/s/gpzm15ipyt01mis/DA_Seg_models.zip?dl=0
unzip download.zip -d ./
```

ps. adding, deleting, editing path could be done at `configs/model/net/gta5_source.yaml`

## Python Environment

We provided 3 ways to setup the environments.

- Using Develop Environment in VS Code (most Recommended)
- Using Docker (Recommended)
- Using Pip/Conda

### Using Develop Environment in VS Code (Recommended)

If you are using VS Code, this is the most recommended way. You can set up all the environment for this project in just one step. 

1. Open this project in VS Code
2. Install extension “Dev Containers”
3. Press Cmd/Ctrl+Shift+P → Dev Container: Rebuild and Reopen in Container

ps. The config folder .devcontainer has been included in our project, you can edit it as if you would introduce or remove some libraries.

ps. Details and Instructions about Dev Container in VS Code can be found here ([link](https://code.visualstudio.com/docs/devcontainers/containers)).

### Using Docker

Dockerfile is at .devcontainer/Dockerfile

### Using Pip/Conda

1. Install pytorch form official website ([link](https://pytorch.org/get-started/locally/)).

```python
pip install \
		tensorboard
		pandas \
    opencv-python \
    pytorch-lightning \
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
    ipdb \
    hydra-joblib-launcher
```

ps. We use pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel while other versions should also work.

# Running the code

### Get Start

The following command will use test the performance of a model pre-trained on `GTA5` with target as the validation set of `Cityscapes`.  Hyper-parameters are set as default.

```python
python src/train.py experiment=ttda
```

Output Example:

```python
wandb: Run summary:
...
wandb: test/acc/dataloaderr13_idx_0 55.01257 # mIoU of 13 class
wandb: test/acc/dataloaderr16_idx_0 49.24151 # mIoU of 16 class
wandb:   test/acc/dataloaderr_idx_0 45.81422 # mIoU of 19 class
...
```

### Custom

You can use the following to start an experiment with custom `source model`, `target dataset` and `hyper-parameters`.

```python
# e.g.
python src/train.py \
	experiment=ttda \
	model/net=gta5_source \
	datamodule/test_list=cityscapes \
	model.cfg.bn_lambda=0.8 \
	model.cfg.proto_lambda=0.8 \
	model.cfg.fusion_lambda=0.8 \
	model.cfg.confidence_threshold=0.9 \
	model.cfg.proto_rho=0.1
```

The available choices are listed as following:

- Source model:

  `model/net=gta5_source,synthia_source,gta5_synthia_source`

- Target dataset:

  `datamodule/test_list=idd,crosscity,cityscapes,bdd,mapillary`

- Hyper-Paramerters:

  bn_lambda: 0-1 (default 0.8)

  proto_lambda: 0-1 (default 0.8)

  fusion_lambda: 0-1 (default 0.8)

  confidence_threshold: 0-1 (default 0.9)

  proto_rho: 0-1 (default 0.1)

ps. File based customization is also supported by modifying `configs/model/diga.yaml` . Note that the cmd line has higher priority if there are conflicting options.
