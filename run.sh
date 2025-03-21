#!/bin/bash
env=generalised-matrax
num_agents=3
python mava/systems_tuning/mat/anakin/mat.py -m env=${env} env.scenario.task_name=matrax-${num_agents}-4-act env.scenario.task_config.num_agents=${num_agents} &&\
python mava/systems_tuning/sable/anakin/ff_sable.py -m env=${env} env.scenario.task_name=matrax-${num_agents}-4-act env.scenario.task_config.num_agents=${num_agents} &&\
python mava/systems_tuning/ppo/anakin/ff_ppo_central.py -m env=${env} env.scenario.task_name=matrax-${num_agents}-4-act env.scenario.task_config.num_agents=${num_agents} &&\
python mava/systems_tuning/ppo/anakin/ff_ppo_central_tabular.py -m env=${env} env.scenario.task_name=matrax-${num_agents}-4-act env.scenario.task_config.num_agents=${num_agents} &&\
python mava/systems_tuning/ppo/anakin/ff_mappo.py -m env=${env} env.scenario.task_name=matrax-${num_agents}-4-act env.scenario.task_config.num_agents=${num_agents} &&\
python mava/systems_tuning/ppo/anakin/ff_ippo.py -m env=${env} env.scenario.task_name=matrax-${num_agents}-4-act env.scenario.task_config.num_agents=${num_agents} &&\
python mava/systems_tuning/ppo/anakin/ff_ppo_central_factored.py -m env=${env} env.scenario.task_name=matrax-${num_agents}-4-act env.scenario.task_config.num_agents=${num_agents} &&\
python mava/systems_tuning/ppo/anakin/ff_ippo_tabular_split.py -m env.scenario.task_name=matrax-${num_agents}-4-act env.scenario.task_config.num_agents=${num_agents}