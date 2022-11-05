python train_manifold.py --dataset cifar10 --outerlayers 24 --innerlayers 20 --hidden_features 100 --outer_image True \
    --latentdim 256 --lr 0.0003 --epoch_recon 50 --epoch_density 70 --batchsize 100
python train_cls.py --dataset cifar10 --num_users 150 --shard_per_user 5 --outerlayers 24 --innerlayers 20 --hidden_features 100 \
    --outer_image True --latentdim 256 --fed_alg local_gen --epochs 200 --local_epochs 15 --batchsize 10 --lr 0.01 --generate_data_weight 0.5 \
    --share_data_weight 0.5 
