#!/bin/bash

export MUJOCO_GL="osmesa"

tasks=(
     $1
)

policies=(
    $2
)

# Replace with your checkpoint directory. This is specified in the hydra config
checkpoint_path="/media/storage/soho/exps/paper/models"
# Replace with your desired output directory.
output_path = "/media/storage/soho/exps/paper/eval"


for task in "${tasks[@]}"; do
    for i in "${!policies[@]}"; do
        echo "Currently runnning training for-> task: $task, policy: ${policies[$i]}"
        if [ ${policies[$i]} = "diffusion_policy" ]; then
            python train.py --config-name=train_diffusion_unet_ddim_image_workspace task.dataset_type=ph task="${task}" training.device=cuda:0 training.seed=1000 
        elif [ ${policies[$i]} = "ibc_policy" ]; then
            python train.py --config-name=train_ibc_hybrid_workspace task.dataset_type=ph task="${task}_image" training.device=cuda:0 training.seed=1000
        elif [[ "${task}" == *"real"* ]]; then
            python train.py --config-name=train_real_fm_unet_image_workspace task.dataset_type=ph policy_type=${policies[$i]} task="${task}" policy.num_inference_steps=11 training.device=cuda:0 +policy.ot_coupling=${ot_coupling[$i]} training.seed=1000
        elif [[ "${task}" == *"metaworld"* ]]; then
            python train.py --config-name=train_cot_unet_metaworld_image_workspace task.dataset_type=ph policy_type=${policies[$i]} task="${task}" policy.num_inference_steps=11 training.device=cuda:0 +policy.ot_coupling=${ot_coupling[$i]} training.seed=1000
        else
            python train.py --config-name=train_cot_unet_image_workspace task.dataset_type=ph policy_type=${policies[$i]} task="${task}" policy.num_inference_steps=11 training.device=cuda:0 +policy.ot_coupling=${ot_coupling[$i]} training.seed=1000
        fi

        echo "Currently runnning evaluation for-> task: $task, policy: ${policies[$i]}, ot_coupling: ${ot_coupling[$i]}"
        # Define arrays for global seeds, environment seeds, and checkpoints
        global_seeds=(10000 20000 30000)  # Replace with your desired global seeds
        env_seeds=(200000 300000 400000)  # Replace with your desired environment seeds
        checkpoints=($(find ${checkpoint_path}/${policies[$i]}_${task}_ph/checkpoints/ -type f -name "epoch*"))
        # Iterate over each checkpoint
        for checkpoint in "${checkpoints[@]}"; do 
            checkpoint_part=$(basename "$(dirname "$(dirname "$checkpoint")")")   
            # Iterate over each global seed
            for global_seed in "${global_seeds[@]}"; do
                # Iterate over each environment seed
                for env_seed in "${env_seeds[@]}"; do
                    # Construct output directory based on checkpoint and seeds
                    output_dir="${output_path}/${checkpoint_part}/gs${global_seed}_es${env_seed}"
                    # Execute the Python script with the specified parameters
                    python eval.py --checkpoint "$checkpoint" -o "$output_dir" -gs "$global_seed" -es "$env_seed"

                done
            done
        done
    done
done