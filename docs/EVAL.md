# Clsoed-Loop Evaluation

To evaluate in the town05long with the 2M checkpoint:
```shell
## In the DriveAdapter/ directory
port_for_carla=22023 ## Change the port for each running script to avoid cofliction
port_for_traffic_manager=22033 ## Change the port for each running script to avoid cofliction
team_agent=driveadapter_agent
is_resume=False ## If there is the corresponding json file in the folder closed_loop_eval_log, you could set it as True to continue after the last finished route.
is_local=True
ckpt_and_config_path=open_loop_training/ckpt/driveadapter_2m.pth+open_loop_training/configs/driveadapter.py
scenario_file=all_towns_traffic_scenarios_no256
cuda_device=0
setting_name=driveadapter_town05long
CUDA_VISIBLE_DEVICES=$cuda_device nohup bash ./leaderboard/scripts/evaluation_town05long.sh $port_for_carla $port_for_traffic_manager $team_agent $is_resume $is_local $ckpt_and_config_path $scenario_file $setting_name 2>&1 > $setting_name.log &
```

or simply:
```shell
## In the DriveAdapter/ directory
CUDA_VISIBLE_DEVICES=0 nohup bash ./leaderboard/scripts/evaluation_town05long.sh 22023 22033 driveadapter_agent  False True open_loop_training/ckpt/driveadapter_2m.pth+open_loop_training/configs/driveadapter.py all_towns_traffic_scenarios_no256 driveadapter_town05long 2>&1 > driveadapter_town05long.log &
```

To evaluate in the longest6 with the 2M checkpoint, you can simply use:
```shell
## In the DriveAdapter/ directory
CUDA_VISIBLE_DEVICES=0 nohup bash ./leaderboard/scripts/evaluation_longest6.sh 23023 23033 driveadapter_agent  False True open_loop_training/ckpt/driveadapter_2m.pth+open_loop_training/configs/driveadapter.py longest6_eval_scenarios driveadapter_longest6 2>&1 > driveadapter_longest6.log &
```

Note that the evaluation result is in the directory **closed_loop_eval_log/results_$setting_name.json** and the visualizations and recordings for debug (top-down view, front view, and canbus) are in the directory **closed_loop_eval_log/eval_log/$setting_name**.

Warning: The visualizations and recordings could take lots of disk space. Please monitor those folders in the [closed_loop_eval_log/eval_log/](../closed_loop_eval_log/eval_log/) and delete those useless ones in time. You could also modify the **save** function of [leaderboard/team_code/driveadapter_agent.py](../leaderboard/team_code/driveadapter_agent.py) to change the saved information during evaluation.

Update: As kindly reminded by the authors of [carla_garge](https://github.com/autonomousvision/carla_garage) (One awsome repo for e2e ad! Check it out for more detials), we update the implementation of longest6 to align with the one used by [Transfuser](https://github.com/autonomousvision/transfuser/tree/2022/leaderboard). Specifically, they ignore the penalty score of running stop sign and increase the number of agents in the scene. DriveAdapter (2M frames) in the updated longest6 achieves DS 63.27, IS 0.87, RC 71.92.