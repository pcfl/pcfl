python train_manifold.py --dataset celeba --outerlayers 24 --innerlayers 20 --hidden_features 100 --outer_image True \
    --latentdim 128 --lr 0.0001 --epoch_recon 50 --epoch_density 70 --batchsize 50
python train_cls.py --dataset celeba --outerlayers 24 --innerlayers 20 --hidden_features 100 \
    --outer_image True --latentdim 128 --fed_alg local_gen --epochs 200 --local_epochs 25 --batchsize 3 --lr 0.01 --generate_data_weight 0.5 \
    --share_data_weight 0.5 
