# GCNDepth
Self-Supervised CNN-GCN Autoencoder 
> **GCNDepth: Self-supervised monocular depth estimation based on graph convolutional network**
>
> To be published ...


## Setup

### Requirements:
- PyTorch1.2+, Python3.5+, Cuda10.0+
- mmcv==0.4.4

Our codes are based on mmcv for distributed learning.


```bash



# this create new conda enviroment for run the model
conda create --name gcndepth python=3.7
conda activate gcndepth

# this installs the right pip and dependencies for the fresh python
conda install ipython
conda install pip

# install required packages from requirements.txt
pip install -r requirements.txt
```

## KITTI training data

Our training data is the same with other self-supervised monocular depth estimation methods, please refer to [monodepth2](https://github.com/nianticlabs/monodepth2) to prepare the training data.

## pretrained weights

We provide weights for [GCNDepth](https://drive.google.com/file/d/1BImXNB9PEgv3mZczB3uBW3EDi4dpOcXF/view?usp=sharing)

## API
We provide an API interface for you to predict depth and pose from an image sequence and visulize some results.
They are stored in folder 'scripts'.

```
eval_pose.py is used to obtain kitti odometry evaluation results.
```

```
eval_depth.py is used to obtain kitti depth evaluation results.
```

```
infer.py is used to generate depth maps from given models.
```

## Training
You can use following command to launch distributed learning of our model:
```
python run.py
```
