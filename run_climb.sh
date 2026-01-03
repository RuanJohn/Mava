#!/bin/bash
env=matrax
python mava/systems/ppo/anakin/ff_ppo_central_autoreg_tabular.py -m env=${env} system.seed=0,1,2,3,4,5,6,7,8,9 &&\
python mava/systems/sable/anakin/ff_sable.py -m env=${env} system.seed=0,1,2,3,4,5,6,7,8,9 &&\
python mava/systems/ppo/anakin/ff_ppo_central.py -m env=${env} system.seed=0,1,2,3,4,5,6,7,8,9 &&\
python mava/systems/ppo/anakin/ff_ppo_central_tabular.py -m env=${env} system.seed=0,1,2,3,4,5,6,7,8,9 &&\
python mava/systems/ppo/anakin/ff_mappo.py -m env=${env} system.seed=0,1,2,3,4,5,6,7,8,9 &&\
python mava/systems/ppo/anakin/ff_ippo.py -m env=${env} system.seed=0,1,2,3,4,5,6,7,8,9 &&\
python mava/systems/ppo/anakin/ff_ppo_central_factored.py -m env=${env} system.seed=0,1,2,3,4,5,6,7,8,9 &&\
python mava/systems/ppo/anakin/ff_ippo_tabular_split.py -m env=${env} system.seed=0,1,2,3,4,5,6,7,8,9 &&\
python mava/systems/ppo/anakin/ff_ppo_central_factored_tabular.py -m env=${env} system.seed=0,1,2,3,4,5,6,7,8,9