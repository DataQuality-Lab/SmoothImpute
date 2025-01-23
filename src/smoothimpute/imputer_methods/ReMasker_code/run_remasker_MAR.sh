# rm -rf ./logs/mean.txt
echo "\n####### running time is $(date) #######\n" >> ./logs/remasker_mar.txt
datasets=("parkinsons" "heart" "libras" "phishing" "bike" "chess" "shuttle")
(
for dataset in "${datasets[@]}"
do
    CUDA_VISIBLE_DEVICES=2 python exp.py --data_name $dataset --missing_mechanism MAR >> ./logs/remasker_mar.txt 2>&1
    echo "Finished processing dataset: $dataset" >> ./logs/remasker_mar.txt
done
) &