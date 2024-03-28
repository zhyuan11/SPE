# Example to run finetuning on NeRF models with 8 views
python run_nerf.py \
    --config configs/freenerf_8v/freenerf_8v_200k_base05.txt \
    --datadir data/nerf_synthetic/ship \
    --expname ship_finenerf_reg0.5_200K
