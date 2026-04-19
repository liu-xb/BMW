python main_v2_weight_res34.py --dataset_target market  --logs log2_trainerv2_weight_market_res34/ --memory_strategy far-update --momentum1 0.9 --momentum2 0.0 --iters 200 --eps 0.7 --batch_size 256 --step 25 --epochs 75 --eval_step 5
python main_v2_weight.py --dataset_target duke  --logs log2_trainerv2_weight_duke/analysis/ --memory_strategy far-update --momentum1 0.7 --momentum2 0.2 --iters 200 --eps 0.7 --batch_size 256 --step 25 --epochs 75 --eval_step 5 
python main_v2_weight.py --dataset_target duke  --logs log2_trainerv2_weight_duke/analysis/ --memory_strategy far-update --momentum1 0.8 --momentum2 0.2 --iters 200 --eps 0.7 --batch_size 256 --step 25 --epochs 75 --eval_step 5 
python main_v2_weight.py --dataset_target duke  --logs log2_trainerv2_weight_duke/analysis/ --memory_strategy far-update --momentum1 1 --momentum2 0.2 --iters 200 --eps 0.7 --batch_size 256 --step 25 --epochs 75 --eval_step 5 
python main_v2_weight.py --dataset_target duke  --logs log2_trainerv2_weight_duke/analysis/ --memory_strategy far-update --momentum1 0.9 --momentum2 0.1 --iters 200 --eps 0.7 --batch_size 256 --step 25 --epochs 75 --eval_step 5 
python main_v2_weight.py --dataset_target duke  --logs log2_trainerv2_weight_duke/analysis/ --memory_strategy far-update --momentum1 0.9 --momentum2 0.15 --iters 200 --eps 0.7 --batch_size 256 --step 25 --epochs 75 --eval_step 5 
python main_v2_weight.py --dataset_target duke  --logs log2_trainerv2_weight_duke/analysis/ --memory_strategy far-update --momentum1 0.9 --momentum2 0.05 --iters 200 --eps 0.7 --batch_size 256 --step 25 --epochs 75 --eval_step 5 
python main_v2_weight.py --dataset_target duke  --logs log2_trainerv2_weight_duke/analysis/ --memory_strategy far-update --momentum1 0.9 --momentum2 0.0 --iters 200 --eps 0.7 --batch_size 256 --step 25 --epochs 75 --eval_step 5 
python main_v2_weight.py --dataset_target duke  --logs log2_trainerv2_weight_duke/analysis/ --memory_strategy far-update --momentum1 0.9 --momentum2 0.25 --iters 200 --eps 0.7 --batch_size 256 --step 25 --epochs 75 --eval_step 5 
python main_v2_weight.py --dataset_target duke  --logs log2_trainerv2_weight_duke/analysis/ --memory_strategy far-update --momentum1 0.9 --momentum2 0.3 --iters 200 --eps 0.7 --batch_size 256 --step 25 --epochs 75 --eval_step 5 
python main_v2_weight.py --dataset_target duke  --logs log2_trainerv2_weight_duke/analysis/ --memory_strategy far-update --momentum1 0.9 --momentum2 0.2 --iters 200 --eps 0.7 --batch_size 256 --step 25 --epochs 75 --eval_step 5 --beta 0.01
python main_v2_weight.py --dataset_target duke  --logs log2_trainerv2_weight_duke/analysis/ --memory_strategy far-update --momentum1 0.9 --momentum2 0.2 --iters 200 --eps 0.7 --batch_size 256 --step 25 --epochs 75 --eval_step 5 --beta 0.03
python main_v2_weight.py --dataset_target duke  --logs log2_trainerv2_weight_duke/analysis/ --memory_strategy far-update --momentum1 0.9 --momentum2 0.2 --iters 200 --eps 0.7 --batch_size 256 --step 25 --epochs 75 --eval_step 5 --beta 0.07
python main_v2_weight.py --dataset_target duke  --logs log2_trainerv2_weight_duke/analysis/ --memory_strategy far-update --momentum1 0.9 --momentum2 0.2 --iters 200 --eps 0.7 --batch_size 256 --step 25 --epochs 75 --eval_step 5 --beta 0.1




# python main_v2_weight.py --dataset_target duke  --logs log2_trainerv2_weight_duke/ --memory_strategy far-update --momentum1 0.9 --momentum2 0.2 --iters 200 --eps 0.7 --batch_size 256 --step 25 --epochs 75 --eval_step 1 --init_path log2_trainerv2_weight_duke/256b-200iter-25step-0.7eps-far-update-0.9-0.2/model_best.pth.tar

# python main_v2_weight.py --dataset_target duke  --logs log2_trainerv2_weight_duke/ --memory_strategy far-update --momentum1 0.9 --momentum2 0.2 --iters 200 --eps 0.6 --batch_size 256 --step 20 --epochs 60 --eval_step 1
# python main_v2_weight.py --dataset_target duke  --logs log2_trainerv2_weight_duke/ --memory_strategy far-update --momentum1 0.9 --momentum2 0.2 --iters 200 --eps 0.6 --batch_size 256 --step 25 --epochs 60 --eval_step 1

# python main_v2_weight.py --dataset_target msmt  --logs log2_trainerv2_weight_msmt/ --memory_strategy far-update --momentum1 0.9 --momentum2 0.0 --iters 400 --eps 0.6 --batch_size 256 --step 20 --epochs 60 --eval_step 5
# python main_v2_weight.py --dataset_target msmt  --logs log2_trainerv2_weight_msmt/ --memory_strategy far-update --momentum1 0.9 --momentum2 0.0 --iters 400 --eps 0.7 --batch_size 256 --step 20 --epochs 60 --eval_step 5
# python main_v2_weight.py --dataset_target msmt  --logs log2_trainerv2_weight_msmt/ --memory_strategy far-update --momentum1 0.9 --momentum2 0.0 --iters 400 --eps 0.8 --batch_size 256 --step 20 --epochs 60 --eval_step 5

# python main_v2_weight.py --dataset_target msmt  --logs log2_trainerv2_weight_msmt/ --memory_strategy far-update --momentum1 0.9 --momentum2 0.2 --init_path log2_trainerv2_weight_msmt/-far-update-0.9-0.2/Epoch30-checkpoint.pth.tar --eval_step 1