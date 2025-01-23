# rm -rf ./logs/mean.txt
echo "\n####### running time is $(date) #######\n" >> ./logs/IGRM_mnar.txt
datasets=("parkinsons" "heart" "libras" "phishing")
(
for dataset in "${datasets[@]}"
do
    CUDA_VISIBLE_DEVICES=3 python main.py --known 0.8 --data $dataset --missing_mechanism MNAR >> ./logs/IGRM_mnar.txt 2>&1
    echo "Finished processing dataset: $dataset" >> ./logs/IGRM_mnar.txt
done
) &