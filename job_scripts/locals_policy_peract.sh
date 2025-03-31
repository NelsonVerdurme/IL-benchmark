export COPPELIASIM_ROOT=${HOME}/robot-3dlotus/dependencies/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT


expr_dir=${HOME}/robot-3dlotus/experiments/18-large
ckpt_step=119000
seed=2025

xvfb-run -a python genrobo3d/evaluation/eval_simple_policy_parrallel.py \
    --expr_dir ${expr_dir} --ckpt_step ${ckpt_step} --num_workers 6 \
    --taskvar_file assets/taskvars_peract.json \
    --seed ${seed} --num_demos 20 \
    --microstep_data_dir /home/huser/data/test/microsteps # --record_video False

