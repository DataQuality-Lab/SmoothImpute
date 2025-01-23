# python train_mdi.py --known 0.7 --log_dir 6 uci --known 0.7 --log_dir 6 uci --data heart
nohup python train_mdi.py --known 0.7 --log_dir 6 uci --data breast  --train_edge 0.7 --train_y 0.7 >> ./logs/breast_03.txt 2>&1 &&
python train_mdi.py --known 0.7 --log_dir 6 uci --data car  --train_edge 0.7 --train_y 0.7 >> ./logs/car_03.txt 2>&1 &&
python train_mdi.py --known 0.7 --log_dir 6 uci --data wireless  --train_edge 0.7 --train_y 0.7 >> ./logs/wireless_03.txt 2>&1 &&
python train_mdi.py --known 0.7 --log_dir 6 uci --data abalone  --train_edge 0.7 --train_y 0.7 >> ./logs/abalone_03.txt 2>&1 &&
python train_mdi.py --known 0.7 --log_dir 6 uci --data turkiye  --train_edge 0.7 --train_y 0.7 >> ./logs/turkiye_03.txt 2>&1 &&
python train_mdi.py --known 0.7 --log_dir 6 uci --data letter  --train_edge 0.7 --train_y 0.7 >> ./logs/letter_03.txt 2>&1 &&
python train_mdi.py --known 0.7 --log_dir 6 uci --data chess  --train_edge 0.7 --train_y 0.7 >> ./logs/chess_03.txt 2>&1 &&
python train_mdi.py --known 0.7 --log_dir 6 uci --data shuttle  --train_edge 0.7 --train_y 0.7 >> ./logs/shuttle_03.txt 2>&1 &