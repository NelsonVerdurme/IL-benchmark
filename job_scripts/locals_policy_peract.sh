export COPPELIASIM_ROOT=${HOME}/mini-diffuse-actor/dependencies/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT


expr_dir=/home/huser/mini-diffuse-actor/experiments/minidiff
ckpt_step=95200

for seed in 0 1 2
do
xvfb-run -a python minidiffuser/evaluation/eval_simple_policy_parrallel.py \
    --expr_dir ${expr_dir} --ckpt_step ${ckpt_step} --num_workers 6 \
    --taskvar_file assets/taskvars_peract.json \
    --seed ${seed} --num_demos 20 \
    --microstep_data_dir /home/huser/data/RLBench-18Task/test/microsteps # --record_video False
done

