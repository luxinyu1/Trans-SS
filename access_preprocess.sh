if [ "$#" == 1 ]; then
    python ./access_preprocess.py --dataset-name $1 \
                            --length-ratio-target-ratio 0.9 \
                            --levenshtein-traget-ratio 0.75 \
                            --word-rank-ratio-traget-ratio 0.75 \
                            --dependency-tree-depth-ratio 0.8 \
                            ;
elif [ "$#" == 0 ]; then
    echo "Please enter the name of dataset!"
else
    echo "You can only preprocess one dataset at a time!"
fi