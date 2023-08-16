python -u train_fp_nn.py --d_hidden_1 0\
    --d_hidden_2 0\
    --d_hidden_3 0\
    --k_folds 10\
    --n_epochs 10 | tee -a log_fp_nn.txt

python -u train_fp_nn.py --d_hidden_1 1796\
    --d_hidden_2 0\
    --d_hidden_3 0\
    --k_folds 10\
    --n_epochs 10 | tee -a log_fp_nn.txt

python -u train_fp_nn.py --d_hidden_1 1880\
    --d_hidden_2 1712\
    --d_hidden_3 0\
    --k_folds 10\
    --n_epochs 10 | tee -a log_fp_nn.txt

python -u train_fp_nn.py --d_hidden_1 1922\
    --d_hidden_2 1796\
    --d_hidden_3 1670\
    --k_folds 10\
    --n_epochs 10 | tee -a log_fp_nn.txt
