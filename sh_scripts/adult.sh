python train_manifold.py --dataset adult --outerlayers 5 --innerlayers 5 --hidden_features 100 --outer_image False \
    --latentdim 32 --lr 0.001 --epoch_recon 70 --epoch_density 70 --batchsize 50
python train_cls.py --dataset adult --outerlayers 5 --innerlayers 5 --hidden_features 100 --outer_image False \
    --latentdim 32 --fed_alg local_gen --epochs 50 --local_epochs 15 --batchsize 1000 --lr 0.01 --generate_data_weight 0.5 \
    --share_data_weight 0.5 
