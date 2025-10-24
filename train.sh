nohup env TQDM_DISABLE=1 python main.py --device 0 --no-use-dyn-c --no-use-sym --checkpoint ./baseline > baseline.log 2>&1 &
nohup env TQDM_DISABLE=1 python main.py --device 1 --use-dyn-c --no-use-sym --checkpoint ./diy_dyn > diy_dyn.log 2>&1 &
nohup env TQDM_DISABLE=1 python main.py --device 2 --no-use-dyn-c --use-sym --checkpoint ./diy_sym > diy_sym.log 2>&1 &
nohup env TQDM_DISABLE=1 python main.py --device 3 --use-dyn-c --use-sym --checkpoint ./diy_dyn_sym > diy_dyn_sym.log 2>&1 &