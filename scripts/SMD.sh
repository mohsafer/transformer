export CUDA_VISIBLE_DEVICES=0,1

 
python main.py --anormly_ratio 1 --num_epochs 2 --batch_size 128 --mode train --dataset SMD --data_path SMD --input_c 38 --output_c 38 --loss_fuc MSE --win_size 105 --patch_size 57 --divergence tsallis --q_param 1.5
python main.py --anormly_ratio 1 --num_epochs 5 --batch_size 128 --mode test --dataset SMD --data_path SMD --input_c 38 --output_c 38 --loss_fuc MSE --win_size 105 --patch_size 57 --divergence tsallis --q_param 1.5


# python main.py --anormly_ratio 0.6 --num_epochs 2 --batch_size 256 --mode train --dataset SMD --data_path SMD --input_c 38 --output_c 38 --loss_fuc MSE --win_size 105 --patch_size 57 --divergence renyi --q_param 1.5
# python main.py --anormly_ratio 0.6 --num_epochs 10 --batch_size 256 --mode test --dataset SMD --data_path SMD --input_c 38 --output_c 38 --loss_fuc MSE --win_size 105 --patch_size 57 --divergence renyi --q_param 1.5

# python main.py --anormly_ratio 0.6 --num_epochs 2 --batch_size 256 --mode train --dataset SMD --data_path SMD --input_c 38 --output_c 38 --loss_fuc MSE --win_size 105 --patch_size 57 --divergence kl --q_param 0.5
# python main.py --anormly_ratio 0.6 --num_epochs 10 --batch_size 256 --mode test --dataset SMD --data_path SMD --input_c 38 --output_c 38 --loss_fuc MSE --win_size 105 --patch_size 57 --divergence kl --q_param 0.5
