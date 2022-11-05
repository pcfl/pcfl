# Precision Collaboration for Federated Learning
the implementation of PCFL.

## introduction for files
* manifold: implementation of the manifold models.
* cls_models: implementation of the the target models.
* dataset: dataset preparation and loading for the models.
* training_manifold/trainer.py: functions about training the manifold models.
* training_cls/trainer.py: functions about training the target models.
* utils/tools.py: needed function for the implementation.
* train_manifold.py: training script of manifold models for all the datasets.
* train_cls.py: training script of target models for all the datasets.

## requirements
python 3.8, pytorch 1.7.1

the needed libraries are in requirements.txt

## dataset generatation:
### CIFAR10
CIFAR10 dataset should be automatically downloaded when running the training script.

### FEMNIST
FEMNIST is from the LEAF repository, we re-sample and re-partition the data as in https://github.com/lgcollins/FedRep.
We modify the sampling script `my_sample.py` to `dataset/femnist_sample.py`.

* download the LEAF repository from https://github.com/TalwalkarLab/leaf/
* move to `leaf-master/data/femnist` and excute the following command to download the raw data:

        ./preprocess.sh -s niid --sf 0.5 -k 50 -tf 0.8 -t sample

* re-sample with our code:

        mv dataset/femnist_sample.py leaf-master/data/femnist/data/
        cd leaf-master/data/femnist/data/
        python femnist_sample.py

### CelebA
We follow the preprocessing steps of https://github.com/litian96/ditto, with the parameter: `sh preprocess.sh -s niid -iu 0.06 --sf 0.06 -k 5 -t sample`

### adult
* download the original adult dataset:
`wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data`  
`wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test`  

* dataset split
`python dataset/adult_generation.py`
the split dataset should be in ./data/adult/train and ./data/adult/test 

### eICU
eICU data set is not directly available and detailed description is in the paper https://arxiv.org/abs/2108.08435

## experiments
the scripts for all the datasets including synthetic experiment are provided in `sh_scripts`.

### parameters
- num_users: the number of clients, used in CIFAR10 dataset and synthetic experiment.
- outerlayers: number of outer layers in the manifold model.
- innerlayers: number of inner layers in the manifold model.
- hidden_features: dimension of some neural network layers in the manifold model.
- lr: learning rate.
- epoch_recon: epoches in the first phase of learning manifold model.
- epoch_density: epoches in the first phase of learning manifold model.
- conditional: whether the manifold model is conditioned on labels.
- share_data_weight: regularization parameter in the loss function, $\alpha$.
- generate_data_weight: regularization parameter in the loss function, $\beta$.
