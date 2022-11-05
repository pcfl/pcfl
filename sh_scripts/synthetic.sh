python train_manifold.py --dataset sinsynthetic --num_users 96 --outerlayers 10 --innerlayers 10 --hidden_features 8 \
    --lr 0.01 --epoch_recon 200 --epoch_density 200 --batchsize 50 --conditional False
python train_cls.py --dataset sinsynthetic --num_users 96 --outerlayers 10 --innerlayers 10 --hidden_features 8 \
    --fed_alg local_gen --epochs 1700 --local_epochs 50 --batchsize 1000 --conditional False --lr 0.1 \
    --generate_data_weight 0.5 --share_data_weight 0.5
