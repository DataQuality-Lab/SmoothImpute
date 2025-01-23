# python train_mdi.py  --known 0.8 --log_dir 5 uci   --known 0.8 --log_dir 5 uci  --data heart
# python train_mdi.py --known 0.8 --log_dir 5 uci  --data breast --train_edge 0.8 --train_y 0.8 >> ./logs/breast_02.txt 2>&1 &&
# python train_mdi.py --known 0.8 --log_dir 5 uci --data car --train_edge 0.8 --train_y 0.8 >> ./logs/car_02.txt 2>&1 &&
# python train_mdi.py --known 0.8 --log_dir 5 uci --data wireless --train_edge 0.8 --train_y 0.8 >> ./logs/wireless_02.txt 2>&1 &&
# python train_mdi.py --known 0.8 --log_dir 5 uci --data abalone --train_edge 0.8 --train_y 0.8 >> ./logs/abalone_02.txt 2>&1 &&
# python train_mdi.py --known 0.8 --log_dir 5 uci --data turkiye --train_edge 0.8 --train_y 0.8 >> ./logs/turkiye_02.txt 2>&1 &&
nohup python train_mdi.py --known 0.8 --log_dir 5 uci --data letter --train_edge 0.8 --train_y 0.8 >> ./logs/letter_02.txt 2>&1 &&
python train_mdi.py --known 0.8 --log_dir 5 uci --data chess --train_edge 0.8 --train_y 0.8 >> ./logs/chess_02.txt 2>&1 &&
python train_mdi.py --known 0.8 --log_dir 5 uci --data shuttle --train_edge 0.8 --train_y 0.8 >> ./logs/shuttle_02.txt 2>&1 &


python train_mdi.py --epochs 2 --known 0.8 --log_dir 5 uci --data wine_nomi --train_edge 0.8 --train_y 0.8 --node_mode 1


python train_mdi.py --known 0.8 --epochs 200 uci --data blood