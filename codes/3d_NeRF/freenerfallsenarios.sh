# Description: Run the FreeNerf model on the Ficus dataset with 8 views and 200k iterations
python run_nerf.py \
    --config configs/freenerf_8v/freenerf_8v_200k_base05.txt \
    --datadir data/nerf_synthetic/ficus \
    --expname ficus_finenerf_reg0.5_200K