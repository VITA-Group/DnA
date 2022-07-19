# get opts
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
# general
  -e|--epochs) epochs="$2"; shift; shift ;;
  -s|--split) split="$2"; shift; shift ;;
  -p|--port) port="$2"; shift; shift ;;
  -w|--workers) workers="$2"; shift; shift ;;
  -g|--GPU_NUM) GPU_NUM=("$2"); shift; shift ;;
  --model) model=("$2"); shift; shift ;;
  --save_dir_prefix) save_dir_prefix=("$2"); shift; shift ;;
# pretrain_checkpoint
  --pretrain_checkpoint) pretrain_checkpoint=("$2"); shift; shift ;;
  --pretrain_checkpoint_name) pretrain_checkpoint_name=("$2"); shift; shift ;;
# small datasets subset
  --few_shot_number) few_shot_number=("$2"); shift; shift ;;
  --few_shot_seed) few_shot_seed=("$2"); shift; shift ;;
# general
  --lr) lr=("$2"); shift; shift ;;
  --wd) wd=("$2"); shift; shift ;;
  --decreasingLr) decreasingLr=("$2"); shift; shift ;;
  --fixBackbone) fixBackbone=("$2"); shift; shift ;;
  --fixBackbone_cnt) fixBackbone_cnt=("$2"); shift; shift ;;
  --batchSize) batchSize=("$2"); shift; shift ;;
  --dataset) dataset=("$2"); shift; shift ;;
  --data) data=("$2"); shift; shift ;;
  --test_freq) test_freq=("$2"); shift; shift ;;
  --late_test) late_test=("$2"); shift; shift ;;
# for simclr
  --simclr) simclr=("$2"); shift; shift ;;
  --tuneFromFirstFC) tuneFromFirstFC=("$2"); shift; shift ;;
  --optimizer) optimizer=("$2"); shift; shift ;;
# for low rank
  --pretrain_low_rank) pretrain_low_rank=("$2"); shift; shift ;;
  --pretrain_low_rank_r_ratio) pretrain_low_rank_r_ratio=("$2"); shift; shift ;;
  --pretrain_low_rank_fix) pretrain_low_rank_fix=("$2"); shift; shift ;;
  --pretrain_low_rank_keep_noise) pretrain_low_rank_keep_noise=("$2"); shift; shift ;;
  --pretrain_low_rank_consecutive) pretrain_low_rank_consecutive=("$2"); shift; shift ;;
  --pretrain_low_rank_lora) pretrain_low_rank_lora=("$2"); shift; shift ;;
  --pretrain_low_rank_merge_to_std_model) pretrain_low_rank_merge_to_std_model=("$2"); shift; shift ;;
# distill
  --skip_tune) skip_tune=("$2"); shift; shift ;;
  --distill) distill=("$2"); shift; shift ;;
  --distill_lr) distill_lr=("$2"); shift; shift ;;
  --distill_temp) distill_temp=("$2"); shift; shift ;;
  --distill_epochs) distill_epochs=("$2"); shift; shift ;;
  --distill_no_ckpt) distill_no_ckpt=("$2"); shift; shift ;;
# visualization
  --out_feature) out_feature=("$2"); shift; shift ;;
  *) echo "${1} is not found"; exit 125;
esac
done

epochs=${epochs:-200}
split=${split:-full}
port=${port:-4833}
workers=${workers:-5}
GPU_NUM=${GPU_NUM:-1}
model=${model:-'res50'}
grace_multi=${grace_multi:-False}
save_dir_prefix=${save_dir_prefix:-"."}

pretrain_checkpoint=${pretrain_checkpoint:-pretrain_weights/moco_v2_800ep_pretrain.pth.tar}
pretrain_checkpoint_name=${pretrain_checkpoint_name:-mocoIN}

few_shot_number=${few_shot_number:-"-1"}
few_shot_seed=${few_shot_seed:-"0"}

lr=${lr:-0.005}
wd=${wd:-None}
decreasingLr=${decreasingLr:-None}
fixBackbone=${fixBackbone:-False}
fixBackbone_cnt=${fixBackbone_cnt:-1}
batchSize=${batchSize:-48}
dataset=${dataset:-CUB200}
data=${data:-CUB200}
test_freq=${test_freq:-10}
late_test=${late_test:-False}
msr=${msr:-False}
msr_single=${msr_single:-False}

simclr=${simclr:-"False"}
tuneFromFirstFC=${tuneFromFirstFC:-"False"}
optimizer=${optimizer:-"sgd"}

pretrain_low_rank=${pretrain_low_rank:-"False"}
pretrain_low_rank_r_ratio=${pretrain_low_rank_r_ratio:-"0.25"}
pretrain_low_rank_keep_noise=${pretrain_low_rank_keep_noise:-"False"}
pretrain_low_rank_consecutive=${pretrain_low_rank_consecutive:-"False"}
pretrain_low_rank_lora=${pretrain_low_rank_lora:-"False"}
pretrain_low_rank_fix=${pretrain_low_rank_fix:-"False"}
low_rank_lambda_s=${low_rank_lambda_s:-"0.01"}
low_rank_compress_step=${low_rank_compress_step:-"1000"}
pretrain_low_rank_merge_to_std_model=${pretrain_low_rank_merge_to_std_model:-"1000"}

skip_tune=${skip_tune:-"False"}
distill=${distill:-"False"}
distill_temp=${distill_temp:-0.1}
distill_lr=${distill_lr:-"2.0"}
distill_epochs=${distill_epochs:-"100"}
distill_no_ckpt=${distill_no_ckpt:-False}

out_feature=${out_feature:-""}

if [[ ${few_shot_number} != "-1" ]]
then
  split="train_sample${few_shot_number}_seed${few_shot_seed}"
  echo "few shot number is ${few_shot_number}, reset split to ${split}"
fi

launch_cmd="python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port ${port}"
save_dir="checkpoints_tune"

save_dir=${save_dir_prefix}/checkpoints_tune

finetune_name=${dataset}_${split}_tune_moco_epoch${epochs}_${optimizer}_lr${lr}

if [[ ${decreasingLr} != "None" ]]
then
  finetune_name="${finetune_name}_d${decreasingLr}"
fi

if [[ ${fixBackbone} == "True" ]]
then
  finetune_name="${finetune_name}_fixB"
  if [[ ${fixBackbone_cnt} != "1" ]]
  then
    finetune_name="${finetune_name}cnt${fixBackbone_cnt}"
  fi
fi

if [[ ${wd} != "None" ]]
then
  finetune_name=${finetune_name}_wd${wd}
fi

if [[ ${tuneFromFirstFC} == "True" ]]
then
  finetune_name="${finetune_name}_turnFromFirstFC"
fi

if [[ ${pretrain_low_rank} == "True" ]]
then
  finetune_name="${finetune_name}_LowRank"
  if [[ ${pretrain_low_rank_merge_to_std_model} == "True" ]]
  then
    finetune_name="${finetune_name}2Std"
  fi
  if [[ ${pretrain_low_rank_fix} == "True" ]]
  then
    finetune_name="${finetune_name}FixSpa"
  fi
fi

if [[ ${pretrain_checkpoint} != "None" ]]
then
  finetune_name=${finetune_name}__${pretrain_checkpoint_name}
fi

cmd="${launch_cmd} finetune.py ${finetune_name} --dataset ${dataset} --data ${data} \
--epochs=${epochs} --test_freq ${test_freq} --save_dir ${save_dir} --optimizer ${optimizer} --lr ${lr} \
--num_workers ${workers} --batch-size ${batchSize}"

if [[ ${out_feature} != '' ]]
then
  cmd="${cmd} --out_feature ${out_feature}"
fi

if [[ ${distill} == "True" ]]
then
  distill_name="${finetune_name}_dtlT${distill_temp}Lr${distill_lr}E${distill_epochs}"
fi

if [[ ${distill_no_ckpt} != "False" ]]
then
  distill_name="${distill_name}scratch"
fi

cmd_distill="${launch_cmd} Naive_Tune/tune.py ${distill_name} --dataset ${dataset} \
--epochs=${distill_epochs} --test_freq ${test_freq} --save_dir ${save_dir} --optimizer ${optimizer} --lr ${distill_lr} \
--num_workers ${workers} --batch-size ${batchSize} --distillation \
--distillation_temp ${distill_temp} --distillation_checkpoint ${save_dir}/${finetune_name}/best_model.pt"

if [[ ${decreasingLr} != "None" ]]
then
  cmd="${cmd} --decreasing_lr ${decreasingLr}"
  cmd_distill="${cmd_distill} --decreasing_lr ${decreasingLr}"
else
  cmd="${cmd} --cosineLr"
  cmd_distill="${cmd_distill} --cosineLr"
fi

if [[ ${fixBackbone} == "True" ]]
then
  cmd="${cmd} --fixBackbone --fixBackbone_cnt ${fixBackbone_cnt}"
fi

if [[ ${wd} != "None" ]]
then
  cmd="${cmd} --wd ${wd}"
  cmd_distill="${cmd_distill} --wd ${wd}"
fi

if [[ ${pretrain_checkpoint} != "None" ]]
then
  cmd="${cmd} --checkpoint_pretrain ${pretrain_checkpoint}"
  if [[ ${distill_no_ckpt} == "False" ]]
  then
    cmd_distill="${cmd_distill} --checkpoint_pretrain ${pretrain_checkpoint}"
  fi
fi

if [[ ${late_test} == "True" ]]
then
  cmd="${cmd} --late_test"
fi

if [[ ${split} != "full" ]]
then
  cmd="${cmd} --customSplit ${split}"
fi

if [[ ${simclr} == "True" ]]
then
  cmd="${cmd} --cvt_state_dict"
  if [[ ${distill_no_ckpt} == "False" ]]
  then
    cmd_distill="${cmd_distill} --cvt_state_dict"
  fi
else
  cmd="${cmd} --cvt_state_dict_moco"
  if [[ ${distill_no_ckpt} == "False" ]]
  then
    cmd_distill="${cmd_distill} --cvt_state_dict_moco"
  fi
fi

if [[ ${tuneFromFirstFC} == "True" ]]
then
  cmd="${cmd} --tuneFromFirstFC"
  if [[ ${distill_no_ckpt} == "False" ]]
  then
    cmd_distill="${cmd_distill} --tuneFromFirstFC"
  fi
fi

if [[ ${pretrain_low_rank} == "True" ]]
then
  append="--pretrain_low_rank --pretrain_low_rank_r_ratio ${pretrain_low_rank_r_ratio} "
  if [[ ${pretrain_low_rank_fix} == "True" ]]
  then
    append="${append} --pretrain_low_rank_fix"
  fi
  if [[ ${pretrain_low_rank_keep_noise} == "True" ]]
  then
    append="${append} --pretrain_low_rank_keep_noise"
  fi
  if [[ ${pretrain_low_rank_consecutive} == "True" ]]
  then
    append="${append} --pretrain_low_rank_consecutive"
  fi
  if [[ ${pretrain_low_rank_lora} == "True" ]]
  then
    append="${append} --pretrain_low_rank_lora"
  fi
  if [[ ${pretrain_low_rank_merge_to_std_model} == "True" ]]
  then
    append="${append} --pretrain_low_rank_merge_to_std_model"
  fi
  cmd="${cmd} ${append}"
  if [[ ${distill_no_ckpt} == "False" ]]
  then
    cmd_distill="${cmd_distill} ${append}"
  fi
fi

if [[ ${skip_tune} != "True" ]]
then
  echo ${cmd}
  ${cmd}
fi

if [[ ${distill} == "True" ]]
then
  echo ${cmd_distill}
  ${cmd_distill}
fi

