#execute train_and_predict.py, either specifying none or 3 arguments like so: python train_and_predict.py "brest-822a-fra-uhslc.csv" "lstm" "np.array([0,1,3,5]).astype('int')" 
#how to call: bash execute_train_and_predict.sh "brest-822a-fra-uhslc.csv" 
for i in $(seq 1 2);
do
    echo $i
    python train_and_predict.py $1 "convlstm" "np.array([5]).astype('int')" 
done