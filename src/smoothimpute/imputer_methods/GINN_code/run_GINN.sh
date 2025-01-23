# rm -rf ./logs/mean.txt
# echo "\n####### running time is $(date) #######\n" >> ./logs/GINN.txt
# datasets=("slump" "iris" "wine" "parkinsons" "heart" "yacht" "ionosphere" "libras" "climate" "credit" "breast" "blood" "raisin" "review" "health" "compression" "phishing" "yeast" "airfoil" "car" "drug" "wireless" "obesity" "abalone" "spam" "turkiye" "bike" "letter" "chess" "news" "shuttle" "connect" "poker_hand" "metro" "power_consumption" "wisdm" "bar")
# (
# for dataset in "${datasets[@]}"
# do
#     CUDA_VISIBLE_DEVICES=2 python exp.py --data_name $dataset >> ./logs/GINN.txt 2>&1
#     echo "Finished processing dataset: $dataset" >> ./logs/GINN.txt
# done
# ) &
echo "\n####### running time is $(date) #######\n" >> ./logs/GINN.txt
datasets=("zoo" "kohkiloyeh" "average")
(
for dataset in "${datasets[@]}"
do
    CUDA_VISIBLE_DEVICES=2 python exp.py --data_name $dataset >> ./logs/GINN_update.txt 2>&1
    echo "Finished processing dataset: $dataset" >> ./logs/GINN_update.txt
done
) &