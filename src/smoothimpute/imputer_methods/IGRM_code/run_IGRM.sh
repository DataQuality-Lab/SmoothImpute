# rm -rf ./logs/mean.txt
echo "\n####### running time is $(date) #######\n" >> ./logs/IGRM.txt
datasets=("slump" "iris" "wine" "parkinsons" "heart" "yacht" "ionosphere" "libras" "climate" "credit" "breast" "blood" "raisin" "review" "health" "compression" "phishing" "yeast" "airfoil" "car" "drug" "wireless" "obesity" "abalone" "spam" "turkiye" "bike" "letter" "chess" "news" "shuttle" "connect" "poker_hand" "metro" "power_consumption" "wisdm" "bar")
(
for dataset in "${datasets[@]}"
do
    CUDA_VISIBLE_DEVICES=6 python main.py --known 0.8 --data $dataset >> ./logs/IGRM.txt 2>&1
    echo "Finished processing dataset: $dataset" >> ./logs/IGRM.txt
done
) &