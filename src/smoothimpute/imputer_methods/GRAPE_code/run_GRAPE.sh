# rm -rf ./logs/mean.txt
# echo "\n####### running time is $(date) #######\n" >> ./logs/grape.txt
# datasets=("slump" "iris" "wine" "parkinsons" "heart" "yacht" "ionosphere" "libras" "climate" "credit" "breast" "blood" "raisin" "review" "health" "compression" "phishing" "yeast" "airfoil" "car" "drug" "wireless" "obesity" "abalone" "spam" "turkiye" "bike" "letter" "chess" "news" "shuttle" "connect" "poker_hand" "metro" "power_consumption" "wisdm" "bar")
# (
# for dataset in "${datasets[@]}"
# do
#     CUDA_VISIBLE_DEVICES=5 python train_mdi.py --known 0.8 uci --data $dataset >> ./logs/grape.txt 2>&1
#     echo "Finished processing dataset: $dataset" >> ./logs/grape.txt
# done
# ) &
echo "\n####### running time is $(date) #######\n" >> ./logs/grape.txt
datasets=("zoo" "kohkiloyeh" "average")
(
for dataset in "${datasets[@]}"
do
    CUDA_VISIBLE_DEVICES=5 python train_mdi.py --known 0.8 uci --data $dataset >> ./logs/grape_update.txt 2>&1
    echo "Finished processing dataset: $dataset" >> ./logs/grape_update.txt
done
) &