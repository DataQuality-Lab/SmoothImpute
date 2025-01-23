nohup python3 -u main.py >> ./logs/all_in_one.txt 2>&1 &


python3 main.py --data housing --domain uci

python main.py --known 0.8 --epochs 2000 --data blood

python main.py --known 0.8 --missing_mechanism MCAR --epochs 2000 --data blood

CUDA_VISIBLE_DEVICES=6 python main.py --known 0.8 --data zoo