expr_dir=data/experiments/peract/3dlotus/v1
ckpt_step=220000


xvfb-run -a python genrobo3d/evaluation/eval_simple_policy_server.py \
    --expr_dir ${expr_dir} --ckpt_step ${ckpt_step} --num_workers 4 \
    --taskvar_file assets/taskvars_peract.json \
    --seed ${seed} --num_demos 20 \
    --microstep_data_dir data/peract/test/microsteps

