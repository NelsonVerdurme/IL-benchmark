expr_dir=${HOME}/robot-3dlotus/experiments/mini-diffuse
ckpt_step=180000
seed=2025

xvfb-run -a python genrobo3d/evaluation/eval_simple_policy_parrallel.py \
    --expr_dir ${expr_dir} --ckpt_step ${ckpt_step} --num_workers 8 \
    --taskvar_file assets/taskvars_peract.json \
    --seed ${seed} --num_demos 20 \
    --microstep_data_dir /home/huser/data/test/microsteps # --record_video False

