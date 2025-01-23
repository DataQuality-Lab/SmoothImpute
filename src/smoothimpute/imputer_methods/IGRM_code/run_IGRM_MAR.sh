echo "\n####### running time is $(date) #######\n" >> ./logs/IGRM_mar.txt
datasets=("parkinsons" "heart" "libras" "phishing")
(
for dataset in "${datasets[@]}"
do
    CUDA_VISIBLE_DEVICES=5 python main.py --known 0.8 --data $dataset --missing_mechanism MAR >> ./logs/IGRM_mar.txt 2>&1
    echo "Finished processing dataset: $dataset" >> ./logs/IGRM_mar.txt
done
) &