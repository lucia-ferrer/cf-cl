#=====================
##  FOR INCRMT CLASS 
#=====================
# MNIST ##
nohup python src/train_pipelines/classic/scratch_classic.py -n SplitMNIST > mnist.out
nohup python src/train_pipelines/classic/naive_classic.py -n SplitMNIST > mnist.out
nohup python src/train_pipelines/classic/replay_classic.py -n SplitMNIST > mnist.out
nohup python src/train_pipelines/classic/ewc_classic.py -n SplitMNIST > mnist.out
nohup python src/train_pipelines/classic/multiband_classic.py -n SplitMNIST > mnist.out

## CIFAR 10 ##
nohup python src/train_pipelines/classic/scratch_classic.py -n SplitCIFAR10 > cifar.out
nohup python src/train_pipelines/classic/naive_classic.py -n SplitCIFAR10 > cifar.out
nohup python src/train_pipelines/classic/replay_classic.py -n SplitCIFAR10 > cifar.out
nohup python src/train_pipelines/classic/ewc_classic.py -n SplitCIFAR10 > cifar.out
nohup python src/train_pipelines/classic/multiband_classic.py -n SplitCIFAR10 > cifar.out

# Mini-ImageNet ##
# nohup python src/train_pipelines/classic/scratch_classic.py -n SplitTinyImageNet  

#=====================
##  FOR NEW INSTANCES 
#=====================
# MNIST ##
nohup python src/train_pipelines/classic/scratch_classic.py -n SplitMNIST -ni  > mnist_ni.out
nohup python src/train_pipelines/classic/naive_classic.py -n SplitMNIST -ni > mnist_ni.out
nohup python src/train_pipelines/classic/replay_classic.py -n SplitMNIST -ni > mnist_ni.out
nohup python src/train_pipelines/classic/ewc_classic.py -n SplitMNIST -ni > mnist_ni.out
nohup python src/train_pipelines/classic/multiband_classic.py -n SplitMNIST -ni > mnist_ni.out

# CIFAR 10 ##
nohup python src/train_pipelines/classic/scratch_classic.py -n SplitCIFAR10 -ni > cifar_ni.out
nohup python src/train_pipelines/classic/naive_classic.py -n SplitCIFAR10 -ni > cifar_ni.out
nohup python src/train_pipelines/classic/replay_classic.py -n SplitCIFAR10 -ni > cifar_ni.out
nohup python src/train_pipelines/classic/ewc_classic.py -n SplitCIFAR10 -ni > cifar_ni.out
nohup python src/train_pipelines/classic/multiband_classic.py -n SplitCIFAR10 -ni > cifar_ni.out

# Mini-ImageNet ##
# nohup python src/train_pipelines/classic/scratch_classic.py -n SplitOmniglot   -ni


#=====================
##  TENSORBOARD 
#=====================
tensorboard --logdir=logs/tb_data
