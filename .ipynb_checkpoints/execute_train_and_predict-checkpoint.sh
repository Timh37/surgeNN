for i in $(seq 1 5);
do
    echo $i
    python train_and_predict.py
done
