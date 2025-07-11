export CUDA_VISIBLE_DEVICES=0

python main.py  --anormly_ratio 1 --num_epochs 1 --batch_size 64 --mode train --dataset MSL --data_path MSL --input_c 55 --output_c 55 --loss_fuc MSE --win_size 105 --patch_size 1357 --divergence tsallis --q_param  0.5
python main.py --anormly_ratio 1 --num_epochs 3 --batch_size 64 --mode test --dataset MSL --data_path MSL --input_c 55 --output_c 55 --loss_fuc MSE --win_size 105 --patch_size 1357 --divergence tsallis --q_param  0.5


# python main.py --anormly_ratio 1 --num_epochs 3   --batch_size 64  --mode train --dataset MSL  --data_path MSL  --input_c 55 --output_c 55  --win_size 90  --patch_size 35
# python main.py --anormly_ratio 1  --num_epochs 10     --batch_size 64    --mode test    --dataset MSL   --data_path MSL --input_c 55    --output_c 55   --win_size 90  --patch_size 35

