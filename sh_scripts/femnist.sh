python train_manifold.py --dataset femnist --outerlayers 5 --innerlayers 5 --hidden_features 100 --outer_image False \
    --latentdim 12 --lr 0.001 --epoch_recon 50 --epoch_density 70 --batchsize 100
python train_cls.py --dataset femnist --outerlayers 5 --innerlayers 5 --hidden_features 100 --outer_image False \
    --latentdim 12 --fed_alg local_gen --epochs 200 --local_epochs 15 --batchsize 10 --lr 0.01 --generate_data_weight 0.5 \
    --share_data_weight 0.5 
