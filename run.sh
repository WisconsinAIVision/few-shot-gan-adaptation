#CUDA_VISIBLE_DEVICES=1,3 python3 patch_trainer.py --batch 4  --data_path ./processed_data/caricatures_dark --exp random_landscapes_p3_f5  --augment --no-ewc --subspace_noise 1 --n_train 10 --highp 3 --subspace_freq 5 --iter 10002 

#CUDA_VISIBLE_DEVICES=0,2 python3 patch_trainer.py --batch 4  --ckpt checkpoints/pretrained_mountains_256/pretrained.pt --data_path ./processed_data/caricatures --exp mountains_caricatures --augment --no-ewc --subspace_noise 1 --n_train 10 --iter 10002  --highp 4 --subspace_freq 2 --source mountains 
#CUDA_VISIBLE_DEVICES=0,2 python3 patch_trainer.py --batch 4  --ckpt checkpoints/pretrained_mountains_256/pretrained.pt --data_path ./processed_data/landscapes_art --exp mountains_landscpaes --augment --no-ewc --subspace_noise 1 --n_train 10 --iter 10002  --highp 4 --subspace_freq 2 --source mountains 
#CUDA_VISIBLE_DEVICES=1,3 python3 patch_trainer.py --batch 4  --ckpt checkpoints/pretrained_mountains_256/pretrained.pt --data_path ./processed_data/haunted --exp mountains_haunted --augment --no-ewc --subspace_noise 1 --n_train 10 --iter 10002  --highp 4 --subspace_freq 2 --source mountains 
#CUDA_VISIBLE_DEVICES=1,3 python3 patch_trainer.py --batch 4  --ckpt checkpoints/pretrained_handsv2_256/pretrained.pt --data_path ./processed_data/fire --exp handsv2_fire --no-augment --no-ewc --subspace_noise 1 --n_train 10 --iter 10002  --highp 4 --save_freq 1000 --subspace_std 0.05 --source handsv2 


#CUDA_VISIBLE_DEVICES=0,2 python3 only_patch_trainer.py --batch 4  --ckpt checkpoints/pretrained_church_256/pretrained.pt --data_path ./processed_data/landscapes_art --exp church_landscapes_only_feat_const  --augment --no-ewc --subspace_noise 0 --n_train 10 --iter 10002  --highp 4 --subspace_freq 2 --source church
#CUDA_VISIBLE_DEVICES=0,2 python3 z_consistency.py --batch 4  --ckpt checkpoints/pretrained_ffhq_256/pretrained.pt --data_path ./processed_data/caricatures --exp ffhq_caricatures_only_feat_const_10_5  --no-augment --no-ewc --iter 5002 --kl_wt 100000 


#CUDA_VISIBLE_DEVICES=0,2 python3 connected_trainer.py --batch 4  --ckpt checkpoints/pretrained_landscapes_256/pretrained.pt --data_path ./processed_data/landscapes_art --exp landscapes_scene_art_line_joint  --augment --no-ewc --subspace_noise 1 --n_train 10 --patch_shape line --iter 20002 --source_key landscapes 

#CUDA_VISIBLE_DEVICES=1,3 python3 connected_trainer.py --batch 4  --ckpt checkpoints/pretrained_horses_256/pretrained.pt --data_path ./processed_data/zebras_processed/ --exp horses_zebras_connected_rect  --augment --no-ewc --subspace_noise 1 --n_train 10 --patch_shape rect --source_key horses


CUDA_VISIBLE_DEVICES=1,3 python3 train.py --batch 4 --ckpt ../work/stylegan2-pytorch/checkpoints/pretrained_ffhq_256/pretrained.pt --data_path ../work/stylegan2-pytorch/processed_data/caricatures --exp ffhq_caricatures_ours_imgs  --augment --n_train 10 --highp 1 --subspace_freq 2 --iter 5002 --img_freq 200 --save_freq 5100 
#CUDA_VISIBLE_DEVICES=0,2 python3 patch_trainer.py --batch 4  --ckpt checkpoints/pretrained_cars_256/pretrained.pt --data_path ./processed_data/wrecked_cars512 --exp cars_wrecked_cars_no_aug  --no-augment --no-ewc --subspace_noise 1 --n_train 8 --highp 3 --subspace_freq 3 --iter 10002 --source_key cars --size 512 --subspace_std 0.05

#CUDA_VISIBLE_DEVICES=0,2 python3 patch_trainer.py --batch 4  --ckpt checkpoints/pretrained_red_noise_3ch_nosp_256/pretrained.pt --data_path ./processed_data/landscapes_art --exp red_noise_nosp_landscapes_ours  --augment --no-ewc --subspace_noise 1 --n_train 10 --highp 4 --subspace_freq 4 --iter 10002 --source_key red_noise


#CUDA_VISIBLE_DEVICES=1,3 python3 freezeD.py --batch 4  --ckpt checkpoints/pretrained_ffhq_256/pretrained.pt --data_path ./processed_data/caricatures/ --exp ffhq_caricatures_freezeD --no-augment --no-ewc --subspace_noise 0


#CUDA_VISIBLE_DEVICES=0,2 python3 train.py --batch 4  --ckpt checkpoints/pretrained_ffhq_256/pretrained.pt --data_path ./processed_data/aligned/Amedeo_Modigliani --exp ffhq_amedeo_simple_ft --augment --no-ewc --subspace_noise 0
#CUDA_VISIBLE_DEVICES=0,2 python3 freezeD.py --batch 4  --ckpt checkpoints/pretrained_ffhq_256/pretrained.pt --data_path ./processed_data/aligned/Amedeo_Modigliani --exp ffhq_amedeo_freezeD --no-augment --no-ewc --subspace_noise 0
