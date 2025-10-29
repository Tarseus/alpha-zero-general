# nohup python grid_search.py --mode othello --cmin-list 0.7 --cmax-list 1.3 --tau-list 16 --danger-list 0.0 --kappa-list 1.0 --games 100 --sims 50 > test2.log 2>&1 &

# nohup python grid_search.py --mode entropy --cmin-list 0.8 --cmax-list 1.3 --games 100 --sims 50 > test_entropy.log 2>&1 &

# cmin=0.7, cmax=1.3, tau=16.0, danger=0.2, kappa=0.8

# nohup python grid_search.py --mode othello --cmin-list "0.6,0.7,0.8" --cmax-list "1.2,1.3,1.4" --tau-list "12,16,24" --danger-list "0,0.1" --kappa-list "0.8,1.0,1.2" --games 100 --sims 50 > test2.log 2>&1 &

nohup python grid_search.py --mode mix --games 100 --sims 50 --cmin-list "0.6" --cmax-list "1.3" --vtau-list "30" --beta-list "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9" > grid_search_mix_50.log 2>&1 &

nohup python grid_search.py --mode mix --games 100 --sims 100 --cmin-list "0.6" --cmax-list "1.3" --vtau-list "30" --beta-list "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9" > grid_search_mix_100.log 2>&1 &
