#!/bin/bash
#SBATCH -Jfashion
#SBATCH --ntasks=1 --ntasks-per-node=10
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-cpu=16G
#SBATCH -t16:00:00
#SBATCH -oReport-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=oupadhyay3@gatech.edu

module load anaconda3/2022.05.0.1 && conda activate /storage/ice-shared/vip-vx4/.conda/envs/llama
cd /storage/ice-shared/vip-vx4/spring2025/vision-safety
yolo task=detect mode=train model=yolov8n.pt data=fashion-dataset/data.yaml epochs=50 imgsz=640 patience=5
