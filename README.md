# CTrGAN


python train.py --name EXP5000 --datafile ./example/configs/train_data.yaml --checkpoints_dir ./example/ --model CTrGAN --ngf 16 --dataset_mode unaligned_sequence --no_dropout --no_flip --loadSize 272 --fineSize 256 --iuv_mode iuv1 --input_nc 4 --output_nc 4 --use_perceptual_loss --pool_size 0 --niter 5 --niter_decay 15 --save_epoch_freq 20 --continue_train --epoch_count 0 --use_sa --use_qsa --seq_len 3 --nThreads 0 --gpu 0


CUDA_VISIBLE_DEVICES=3 python train.py --name EXP5001 --datafile ./example/configs/train_data.yaml --checkpoints_dir ./example/ --model CTrGAN --ngf 16 --dataset_mode unaligned_sequence --no_dropout --no_flip --loadSize 272 --fineSize 256 --iuv_mode iuv1 --input_nc 4 --output_nc 4 --use_perceptual_loss --pool_size 0 --niter 20 --niter_decay 60 --save_epoch_freq 20 --continue_train --epoch_count 0 --use_sa --use_qsa --seq_len 3 --nThreads 0 --gpu 0

CUDA_VISIBLE_DEVICES=3 python train.py --name EXP5002 --datafile ./example/configs/train_data.yaml --checkpoints_dir ./example/ --model CTrGAN --ngf 16 --dataset_mode unaligned_sequence --no_dropout --no_flip --loadSize 272 --fineSize 256 --iuv_mode iuv1 --input_nc 4 --output_nc 4 --use_perceptual_loss --pool_size 0 --niter 100 --niter_decay 300 --save_epoch_freq 20 --continue_train --epoch_count 0 --use_sa --use_qsa --seq_len 3 --nThreads 0 --gpu 0


python3 --name EXP5001 --datafile ./example/configs/valid_data.yaml --checkpoints_dir ./example/ --model CTrGAN --ngf 16 --dataset_mode unaligned_sequence --no_dropout --no_flip --loadSize 256 --fineSize 256 --iuv_mode iuv1 --input_nc 4 --output_nc 4 --which_epoch 80 --results_dir ./example/results/ --use_fullseq --seq_len 3 --use_sa --use_qsa --gpu 0



CUDA_VISIBLE_DEVICES=3 python train.py  --name EXP5010 --datafile ./example/configs/train_data.yaml --checkpoints_dir ./example/ --model CTrGAN --ngf 16 --dataset_mode unaligned_sequence --no_dropout --no_flip --loadSize 272 --fineSize 256 --iuv_mode iuv1 --input_nc 4 --output_nc 4 --use_perceptual_loss --pool_size 0 --niter 20 --niter_decay 60 --save_epoch_freq 20 --continue_train --epoch_count 0 --use_sa --use_qsa --seq_len 3 --nThreads 4 --gpu 0
CUDA_VISIBLE_DEVICES=2 python train.py  --name EXP5011 --datafile ./example/configs/train_data2.yaml --checkpoints_dir ./example/ --model CTrGAN --ngf 16 --dataset_mode unaligned_sequence --no_dropout --no_flip --loadSize 272 --fineSize 256 --iuv_mode iuv1 --input_nc 4 --output_nc 4 --use_perceptual_loss --pool_size 0 --niter 20 --niter_decay 60 --save_epoch_freq 20 --continue_train --epoch_count 0 --use_sa --use_qsa --seq_len 3 --nThreads 4 --gpu 0
