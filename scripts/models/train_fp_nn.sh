python -u train_fp_nn.py --d_hidden_1 0\
    --d_hidden_2 0\
    --d_hidden_3 0\
    --k_folds 5\
    --n_epochs 10 | tee -a log_fp_nn.txt

python -u train_fp_nn.py --d_hidden_1 1796\
    --d_hidden_2 0\
    --d_hidden_3 0\
    --k_folds 5\
    --n_epochs 10 | tee -a log_fp_nn.txt

python -u train_fp_nn.py --d_hidden_1 1880\
    --d_hidden_2 1712\
    --d_hidden_3 0\
    --k_folds 5\
    --n_epochs 10 | tee -a log_fp_nn.txt

python -u train_fp_nn.py --d_hidden_1 1922\
    --d_hidden_2 1796\
    --d_hidden_3 1670\
    --k_folds 5\
    --n_epochs 10 | tee -a log_fp_nn.txt

python -u train_fp_nn.py --d_hidden_1 2048\
    --d_hidden_2 0\
    --d_hidden_3 0\
    --k_folds 5\
    --n_epochs 10 | tee -a log_fp_nn.txt

python -u train_fp_nn.py --d_hidden_1 3072\
    --d_hidden_2 0\
    --d_hidden_3 0\
    --k_folds 5\
    --n_epochs 10 | tee -a log_fp_nn.txt


python -u train_fp_nn.py --d_hidden_1 4096\
    --d_hidden_2 0\
    --d_hidden_3 0\
    --k_folds 5\
    --n_epochs 10 | tee -a log_fp_nn.txt

python -u train_fp_nn.py --d_hidden_1 4096\
    --d_hidden_2 2048\
    --d_hidden_3 0\
    --k_folds 5\
    --n_epochs 10 | tee -a log_fp_nn.txt

python -u train_fp_nn.py --d_hidden_1 4096\
    --d_hidden_2 3072\
    --d_hidden_3 2048\
    --k_folds 5\
    --n_epochs 10 | tee -a log_fp_nn.txt

python -u train_fp_nn.py --d_hidden_1 1544\
    --d_hidden_2 0\
    --d_hidden_3 0\
    --k_folds 5\
    --n_epochs 10 | tee -a log_fp_nn.txt

python -u train_fp_nn.py --d_hidden_1 1024\
    --d_hidden_2 0\
    --d_hidden_3 0\
    --k_folds 5\
    --n_epochs 10 | tee -a log_fp_nn.txt

python -u train_fp_nn.py --d_hidden_1 1024\
    --d_hidden_2 512\
    --d_hidden_3 0\
    --k_folds 5\
    --n_epochs 10 | tee -a log_fp_nn.txt

python -u train_fp_nn.py --d_hidden_1 1024\
    --d_hidden_2 512\
    --d_hidden_3 256\
    --k_folds 5\
    --n_epochs 10 | tee -a log_fp_nn.txt

python -u train_fp_nn.py --d_hidden_1 512\
    --d_hidden_2 0\
    --d_hidden_3 0\
    --k_folds 5\
    --n_epochs 10 | tee -a log_fp_nn.txt

python -u train_fp_nn.py --d_hidden_1 512\
    --d_hidden_2 256\
    --d_hidden_3 0\
    --k_folds 5\
    --n_epochs 10 | tee -a log_fp_nn.txt

python -u train_fp_nn.py --d_hidden_1 256\
    --d_hidden_2 0\
    --d_hidden_3 0\
    --k_folds 5\
    --n_epochs 10 | tee -a log_fp_nn.txt

python -u train_fp_nn.py --d_hidden_1 256\
    --d_hidden_2 128\
    --d_hidden_3 0\
    --k_folds 5\
    --n_epochs 10 | tee -a log_fp_nn.txt

python -u train_fp_nn.py --d_hidden_1 128\
    --d_hidden_2 0\
    --d_hidden_3 0\
    --k_folds 5\
    --n_epochs 10 | tee -a log_fp_nn.txt
