#!/bin/bash
#SBATCH --job-name=05_create_summary_index_ucf_crime_lavad
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-0%1
#SBATCH --output=output/05_create_summary_index_ucf_crime_%A_%a.out
#SBATCH --mail-user=newbiechd@outlook.com
#SBATCH --mail-type=ALL

# Set the UCF Crime directory
ucf_crime_dir="/home/JJ_Group/changhd2504/lavad/datasets/ucf_crime"

# Set paths
root_path="${ucf_crime_dir}/frames"
annotationfile_path="${ucf_crime_dir}/annotations/test.txt"
llm_model_name="llama-2-13b-chat"
batch_size=64
frame_interval=16
index_dim=1024
index_name="opt-6.7b-coco+opt-6.7b+flan-t5-xxl+flan-t5-xl+flan-t5-xl-coco"

# Activate the virtual environment
# VENV_DIR="/path/to/venv/lavad"
# shellcheck source=/dev/null
# source "$VENV_DIR/bin/activate"
# conda activate lavad

captions_dir="${ucf_crime_dir}/captions/summary/${llm_model_name}/${index_name}/"
output_dir="${ucf_crime_dir}/index/summary/${llm_model_name}/${index_name}/index_flat_ip/"
python -m src.models.create_summary_index \
    --index_dim "$index_dim" \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --captions_dir "${captions_dir}" \
    --output_dir "${output_dir}"
