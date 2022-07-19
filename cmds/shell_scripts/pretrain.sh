# get opts
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
# general
  -p|--port) port="$2"; shift; shift ;;
  -w|--workers) workers="$2"; shift; shift ;;
  -g|--GPU_NUM) GPU_NUM=("$2"); shift; shift ;;
  --batchSize) batchSize=("$2"); shift; shift ;;
  --dataset) dataset=("$2"); shift; shift ;;
  --data) data=("$2"); shift; shift ;;
  --save_freq) save_freq=("$2"); shift; shift ;;
  --name_append) name_append=("$2"); shift; shift ;;
  --save_dir_prefix) save_dir_prefix=("$2"); shift; shift ;;
# pretrain
  --pretrain_epochs) pretrain_epochs="$2"; shift; shift ;;
  --pretrain_split) pretrain_split="$2"; shift; shift ;;
# for simclr
  --lr) lr=("$2"); shift; shift ;;
  --temp) temp=("$2"); shift; shift ;;
  --pretrain_checkpoint) pretrain_checkpoint=("$2"); shift; shift ;;
  --pretrain_checkpoint_name) pretrain_checkpoint_name=("$2"); shift; shift ;;
  --aug_strength) aug_strength=("$2"); shift; shift ;;
# low rank
  --low_rank) low_rank=("$2"); shift; shift ;;
  --low_rank_r_ratio) low_rank_r_ratio=("$2"); shift; shift ;;
  --low_rank_lambda_s) low_rank_lambda_s=("$2"); shift; shift ;;
  --low_rank_compress_step) low_rank_compress_step=("$2"); shift; shift ;;
  --low_rank_fix_sparse) low_rank_fix_sparse=("$2"); shift; shift ;;
  --low_rank_fix_low_rank) low_rank_fix_low_rank=("$2"); shift; shift ;;
  --low_rank_tune_U) low_rank_tune_U=("$2"); shift; shift ;;
  --low_rank_tune_V) low_rank_tune_V=("$2"); shift; shift ;;
  --low_rank_tune_U_S) low_rank_tune_U_S=("$2"); shift; shift ;;
  --low_rank_tune_V_S) low_rank_tune_V_S=("$2"); shift; shift ;;
  --low_rank_tune_all) low_rank_tune_all=("$2"); shift; shift ;;
  --low_rank_only_decompose) low_rank_only_decompose=("$2"); shift; shift ;;
  --low_rank_keep_noise) low_rank_keep_noise=("$2"); shift; shift ;;
  --low_rank_UV_lr_ratio) low_rank_UV_lr_ratio=("$2"); shift; shift ;;
  --low_rank_consecutive) low_rank_consecutive=("$2"); shift; shift ;;
  --low_rank_decompose_no_s) low_rank_decompose_no_s=("$2"); shift; shift ;;
  --low_rank_lora_mode) low_rank_lora_mode=("$2"); shift; shift ;;
  --low_rank_sparse_ratio) low_rank_sparse_ratio=("$2"); shift; shift ;;
  *) echo "${1} is not found"; exit 125;
esac
done

port=${port:-4833}
workers=${workers:-10}
GPU_NUM=${GPU_NUM:-1}
batchSize=${batchSize:-256}
dataset=${dataset:-"cifar100"}
data=${data:-"placeholder"}
save_freq=${save_freq:-100}
name_append=${name_append:-""}
save_dir_prefix=${save_dir_prefix:-"."}

pretrain_epochs=${pretrain_epochs:-200}
pretrain_split=${pretrain_split:-full}

lr=${lr:-5.0}
temp=${temp:-0.2}
pretrain_checkpoint_name=${pretrain_checkpoint_name:-None}
pretrain_checkpoint=${pretrain_checkpoint:-None}
aug_strength=${aug_strength:-2.0}

low_rank=${low_rank:-"False"}
low_rank_r_ratio=${low_rank_r_ratio:-"0.25"}
low_rank_compress_step=${low_rank_compress_step:-"50"}
low_rank_lambda_s=${low_rank_lambda_s:-"0.01"}
low_rank_fix_sparse=${low_rank_fix_sparse:-"False"}
low_rank_fix_low_rank=${low_rank_fix_low_rank:-"False"}
low_rank_tune_V=${low_rank_tune_V:-"False"}
low_rank_tune_U=${low_rank_tune_U:-"False"}
low_rank_tune_V_S=${low_rank_tune_V_S:-"False"}
low_rank_tune_U_S=${low_rank_tune_U_S:-"False"}
low_rank_tune_all=${low_rank_tune_all:-"False"}
low_rank_only_decompose=${low_rank_only_decompose:-"False"}
low_rank_keep_noise=${low_rank_keep_noise:-"False"}
low_rank_UV_lr_ratio=${low_rank_UV_lr_ratio:-"1"}
low_rank_consecutive=${low_rank_consecutive:-"False"}
low_rank_decompose_no_s=${low_rank_decompose_no_s:-"False"}
low_rank_lora_mode=${low_rank_lora_mode:-"False"}
low_rank_sparse_ratio=${low_rank_sparse_ratio:-"-1"}

launch_cmd="python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port ${port}"
save_dir=${save_dir_prefix}/checkpoints_moco

pretrain_name=SimCLR_${dataset}_${pretrain_split}_epoch${pretrain_epochs}_lr${lr}_b${batchSize}_temp${temp}_stren${aug_strength}


if [[ ${low_rank} == "True" ]]
then
  pretrain_name="${pretrain_name}_LowRankR${low_rank_r_ratio}Step${low_rank_compress_step}s${low_rank_lambda_s}"
  if [[ ${low_rank_sparse_ratio} != "-1" ]]
  then
    pretrain_name="${pretrain_name}FS${low_rank_sparse_ratio}"
  fi
  if [[ ${low_rank_fix_sparse} == "True" ]]
  then
    pretrain_name="${pretrain_name}FixSpa"
  fi
  if [[ ${low_rank_fix_low_rank} == "True" ]]
  then
    pretrain_name="${pretrain_name}FixLowR"
  fi
  if [[ ${low_rank_tune_U} == "True" ]]
  then
    pretrain_name="${pretrain_name}TuneU"
  fi
  if [[ ${low_rank_tune_V} == "True" ]]
  then
    pretrain_name="${pretrain_name}TuneV"
  fi
  if [[ ${low_rank_tune_U_S} == "True" ]]
  then
    pretrain_name="${pretrain_name}TuneUS"
  fi
  if [[ ${low_rank_tune_V_S} == "True" ]]
  then
    pretrain_name="${pretrain_name}TuneVS"
  fi
  if [[ ${low_rank_tune_all} == "True" ]]
  then
    pretrain_name="${pretrain_name}TuneAll"
  fi
  if [[ ${low_rank_UV_lr_ratio} != "1" ]]
  then
    pretrain_name="${pretrain_name}UVlrR${low_rank_UV_lr_ratio}"
  fi
  if [[ ${low_rank_consecutive} == "True" ]]
  then
    pretrain_name="${pretrain_name}Wconse"
  fi
  if [[ ${low_rank_decompose_no_s} == "True" ]]
  then
    pretrain_name="${pretrain_name}NoS"
  fi
  if [[ ${low_rank_keep_noise} == "True" ]]
  then
    pretrain_name="${pretrain_name}wNoise"
  fi
  if [[ ${low_rank_lora_mode} == "True" ]]
  then
    pretrain_name="${pretrain_name}LoRA"
  fi
  if [[ ${pretrain_checkpoint} == "None" ]]
  then
    echo "pretraining model need to be True for DnA"; exit 125;
  fi
fi

if [[ ${pretrain_checkpoint} != "None" ]]
then
  pretrain_name="${pretrain_name}__${pretrain_checkpoint_name}"
fi

if [[ ${name_append} != "" ]]
then
  pretrain_name="${pretrain_name}_${name_append}"
fi

cmd_pretrain="${launch_cmd} train_simclr.py ${pretrain_name} --epochs=${pretrain_epochs} --dataset ${dataset} --data ${data} \
 --mlp --moco-t 0.2 --aug-plus --cos --optimizer lars --wd 1e-6 --simclr-t ${temp} \
 --color-jitter-strength ${aug_strength} --lr ${lr} -b ${batchSize}  \
 --workers ${workers} --save_dir ${save_dir} --resume --save_freq ${save_freq}"

if [[ ${pretrain_split} != "full" ]]
then
  cmd_pretrain="${cmd_pretrain} --customSplit ${pretrain_split}"
fi

if [[ ${pretrain_checkpoint} != "None" ]]
then
  cmd_pretrain="${cmd_pretrain} --checkpoint_pretrain ${pretrain_checkpoint}"
fi

if [[ ${low_rank} == "True" ]]
then
  cmd_pretrain="${cmd_pretrain} --low_rank --low_rank_r_ratio ${low_rank_r_ratio} \
  --low_rank_lambda_s ${low_rank_lambda_s} --low_rank_compress_step ${low_rank_compress_step} \
  --low_rank_UV_lr_ratio ${low_rank_UV_lr_ratio}"
  if [[ ${low_rank_fix_sparse} == "True" ]]
  then
    cmd_pretrain="${cmd_pretrain} --low_rank_fix_sparse"
  fi
  if [[ ${low_rank_fix_low_rank} == "True" ]]
  then
    cmd_pretrain="${cmd_pretrain} --low_rank_fix_low_rank"
  fi
  if [[ ${low_rank_tune_U} == "True" ]]
  then
    cmd_pretrain="${cmd_pretrain} --low_rank_tune_U"
  fi
  if [[ ${low_rank_tune_V} == "True" ]]
  then
    cmd_pretrain="${cmd_pretrain} --low_rank_tune_V"
  fi
  if [[ ${low_rank_tune_U_S} == "True" ]]
  then
    cmd_pretrain="${cmd_pretrain} --low_rank_tune_U_S"
  fi
  if [[ ${low_rank_tune_V_S} == "True" ]]
  then
    cmd_pretrain="${cmd_pretrain} --low_rank_tune_V_S"
  fi
  if [[ ${low_rank_tune_all} == "True" ]]
  then
    cmd_pretrain="${cmd_pretrain} --low_rank_tune_all"
  fi
  if [[ ${low_rank_only_decompose} == "True" ]]
  then
    cmd_pretrain="${cmd_pretrain} --low_rank_only_decompose"
  fi
  if [[ ${low_rank_keep_noise} == "True" ]]
  then
    cmd_pretrain="${cmd_pretrain} --low_rank_keep_noise"
  fi
  if [[ ${low_rank_consecutive} == "True" ]]
  then
    cmd_pretrain="${cmd_pretrain} --low_rank_reshape_consecutive"
  fi
  if [[ ${low_rank_decompose_no_s} == "True" ]]
  then
    cmd_pretrain="${cmd_pretrain} --low_rank_decompose_no_s"
  fi
  if [[ ${low_rank_lora_mode} == "True" ]]
  then
    cmd_pretrain="${cmd_pretrain} --low_rank_lora_mode"
  fi
  if [[ ${low_rank_sparse_ratio} != "-1" ]]
  then
    cmd_pretrain="${cmd_pretrain} --low_rank_sparse_ratio ${low_rank_sparse_ratio}"
  fi
fi

####################### run #########################
echo ${cmd_pretrain}
${cmd_pretrain}
