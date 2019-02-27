# DenseNets-Pytoch
#### Densely Connected Convolutional Networks (DenseNets) in pytorch

This is a PyTorch implementation of [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) paper.

 
## Requirements

- [python3.6](http://www.python.org/)
- [PyTorch](http://pytorch.org/)
- [numpy](http://www.numpy.org/)
- [argparse](https://github.com/python/cpython/blob/3.7/Lib/argparse.py)
- [tensorboardX](https://github.com/lanpa/tensorboardX)

## Usage:

## Prepare
```
CUDA_VISIBLE_DEVICES=0
```

## Training
Example usage for (DenseNet-40-12):
```
python train.py --layers 40 --growth 12 --no-bottleneck --reduce 1.0 --name DenseNet-40-12
```

## TensorboardX Graph

```
tensorboard --logdir runs
```
Open in browser [http://localhost:6006](http://localhost:6006)


