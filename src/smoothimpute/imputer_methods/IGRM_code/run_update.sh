# rm -rf ./logs/mean.txt
echo "\n####### running time is $(date) #######\n" >> ./logs/IGRM.txt
datasets=("zoo" "kohkiloyeh" "average")
(
for dataset in "${datasets[@]}"
do
    CUDA_VISIBLE_DEVICES=6 python main.py --known 0.8 --data $dataset >> ./logs/IGRM_update.txt 2>&1
    echo "Finished processing dataset: $dataset" >> ./logs/IGRM_update.txt
done
) &