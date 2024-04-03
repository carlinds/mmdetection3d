#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 [--name <name>] [--config <config>] [--data-train <data_train>] [--data-val <data_val>] [--sweep-input <sweep_input>] [--cfg-options <cfg_options>]"
    exit 1
}

# Set default values
name=""
config=""
data_root_train=""
data_root_val=""
sweep_input=""
cfg_options=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --name)
            name="$2"
            shift
            ;;
        --config)
            config="$2"
            shift
            ;;
        --data-train)
            data_root_train="$2"
            shift
            ;;
        --data-val)
            data_root_val="$2"
            shift
            ;;
        --sweep-input)
            sweep_input="$2"
            shift
            ;;
        --cfg-options)
            cfg_options="$2"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown parameter passed: $1"
            usage
            ;;
    esac
    shift
done

# Split the input string into parameter name and values
IFS='=' read -r param_name values <<< "$sweep_input"
# Split the values into an array
IFS=',' read -r -a values_array <<< "$values"

for value in "${values_array[@]}"
do
    # Run the training script with the given parameters
    sweep_name="${param_name}_${value}"
    WANDB_GROUP=$name sbatch scripts/slurm_train_berz_single_gpu.sh $sweep_name $config $data_root_train $data_root_val $param_name=$value $cfg_options
done
