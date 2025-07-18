export CUDA_VISIBLE_DEVICES=0

python main.py  --anormly_ratio 4 --num_epochs 1 --batch_size 128 --mode train --dataset PSM --data_path PSM --input_c 25 --output_c 25 --loss_fuc MSE --win_size 105 --patch_size 1 --divergence tsallis --q_param  0.5
python main.py  --anormly_ratio 4 --num_epochs 3 --batch_size 128 --mode test --dataset PSM --data_path PSM --input_c 25 --output_c 25 --loss_fuc MSE --win_size 105 --patch_size 1 --divergence tsallis --q_param 0.5

# python main.py --anormly_ratio 1 --num_epochs 3    --batch_size 128  --mode train --dataset PSM  --data_path PSM --input_c 25    --output_c 25  --loss_fuc MSE  --win_size 60  --patch_size 135
# python main.py --anormly_ratio 1  --num_epochs 10       --batch_size 128     --mode test    --dataset PSM   --data_path PSM  --input_c 25    --output_c 25  --loss_fuc MSE  --win_size 60  --patch_size 1350