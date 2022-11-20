import os

# MATCHER FOR GET FINAL AVERAGE CCR

# python3 rbf.py -t ../datasetsLA3IMC/csv/train_nomnist.csv -T ../datasetsLA3IMC/csv/test_nomnist.csv -c -r 0.01 | grep "Test CCR" | grep -oP '(?<=:).*' | grep "+-" | sed 's/.$//'

# MATCHER FOR GET FINAL AVERAGE MSE

# python3 rbf.py -t ../datasetsLA3IMC/csv/train_nomnist.csv -T ../datasetsLA3IMC/csv/test_nomnist.csv -c -r 0.01 | grep "Test MSE" | grep -oP '(?<=:).*' | grep "+-" | sed 's/.$//'


# EXPERIMENT 1
os.system("echo 'EXPERIMENT 1 (ratios)' >> experiments_results.out")

os.system("echo 'SIN DATASET' >> experiments_results.out")

os.system('python3 rbf.py -t ../datasetsLA3IMC/csv/train_sin.csv -T ../datasetsLA3IMC/csv/test_sin.csv -r 0.05 | grep "Test MSE" | grep -oP "(?<=:).*" | grep "+-" | sed "s/.$//" >> experiments_results.out')

# Do the same for r 0.15, 0.25 and 0.5 and train_sin.csv

os.system('python3 rbf.py -t ../datasetsLA3IMC/csv/train_sin.csv -T ../datasetsLA3IMC/csv/test_sin.csv -r 0.15 | grep "Test MSE" | grep -oP "(?<=:).*" | grep "+-" | sed "s/.$//" >> experiments_results.out')

os.system('python3 rbf.py -t ../datasetsLA3IMC/csv/train_sin.csv -T ../datasetsLA3IMC/csv/test_sin.csv -r 0.25 | grep "Test MSE" | grep -oP "(?<=:).*" | grep "+-" | sed "s/.$//" >> experiments_results.out')

os.system('python3 rbf.py -t ../datasetsLA3IMC/csv/train_sin.csv -T ../datasetsLA3IMC/csv/test_sin.csv -r 0.5 | grep "Test MSE" | grep -oP "(?<=:).*" | grep "+-" | sed "s/.$//" >> experiments_results.out')

# EXPERIMENT 2

os.system("echo 'QUAKE DATASET' >> experiments_results.out")

os.system('python3 rbf.py -t ../datasetsLA3IMC/csv/train_quake.csv -T ../datasetsLA3IMC/csv/test_quake.csv -r 0.05 | grep "Test MSE" | grep -oP "(?<=:).*" | grep "+-" | sed "s/.$//" >> experiments_results.out')

# Do the same for r 0.15, 0.25 and 0.5 and train_quake.csv

os.system('python3 rbf.py -t ../datasetsLA3IMC/csv/train_quake.csv -T ../datasetsLA3IMC/csv/test_quake.csv -r 0.15 | grep "Test MSE" | grep -oP "(?<=:).*" | grep "+-" | sed "s/.$//" >> experiments_results.out')

os.system('python3 rbf.py -t ../datasetsLA3IMC/csv/train_quake.csv -T ../datasetsLA3IMC/csv/test_quake.csv -r 0.25 | grep "Test MSE" | grep -oP "(?<=:).*" | grep "+-" | sed "s/.$//" >> experiments_results.out')

os.system('python3 rbf.py -t ../datasetsLA3IMC/csv/train_quake.csv -T ../datasetsLA3IMC/csv/test_quake.csv -r 0.5 | grep "Test MSE" | grep -oP "(?<=:).*" | grep "+-" | sed "s/.$//" >> experiments_results.out')

# EXPERIMENT 3

os.system("echo 'PARKINSONS DATASET' >> experiments_results.out")

os.system('python3 rbf.py -t ../datasetsLA3IMC/csv/train_parkinsons.csv -T ../datasetsLA3IMC/csv/test_parkinsons.csv -r 0.05 | grep "Test MSE" | grep -oP "(?<=:).*" | grep "+-" | sed "s/.$//" >> experiments_results.out')

# Do the same for r 0.15, 0.25 and 0.5 and train_parkinsons.csv

os.system('python3 rbf.py -t ../datasetsLA3IMC/csv/train_parkinsons.csv -T ../datasetsLA3IMC/csv/test_parkinsons.csv -r 0.15 | grep "Test MSE" | grep -oP "(?<=:).*" | grep "+-" | sed "s/.$//" >> experiments_results.out')

os.system('python3 rbf.py -t ../datasetsLA3IMC/csv/train_parkinsons.csv -T ../datasetsLA3IMC/csv/test_parkinsons.csv -r 0.25 | grep "Test MSE" | grep -oP "(?<=:).*" | grep "+-" | sed "s/.$//" >> experiments_results.out')

os.system('python3 rbf.py -t ../datasetsLA3IMC/csv/train_parkinsons.csv -T ../datasetsLA3IMC/csv/test_parkinsons.csv -r 0.5 | grep "Test MSE" | grep -oP "(?<=:).*" | grep "+-" | sed "s/.$//" >> experiments_results.out')


# EXPERIMENT 4

os.system("echo 'ILDP DATASET' >> experiments_results.out")

os.system('python3 rbf.py -t ../datasetsLA3IMC/csv/train_ildp.csv -T ../datasetsLA3IMC/csv/test_ildp.csv -r 0.05 -c | grep "Test CCR" | grep -oP "(?<=:).*" | grep "+-" | sed "s/.$//" >> experiments_results.out')

# Do the same for r 0.15, 0.25 and 0.5 and train_ildp.csv

os.system('python3 rbf.py -t ../datasetsLA3IMC/csv/train_ildp.csv -T ../datasetsLA3IMC/csv/test_ildp.csv -r 0.15 -c | grep "Test CCR" | grep -oP "(?<=:).*" | grep "+-" | sed "s/.$//" >> experiments_results.out')

os.system('python3 rbf.py -t ../datasetsLA3IMC/csv/train_ildp.csv -T ../datasetsLA3IMC/csv/test_ildp.csv -r 0.25 -c | grep "Test CCR" | grep -oP "(?<=:).*" | grep "+-" | sed "s/.$//" >> experiments_results.out')

os.system('python3 rbf.py -t ../datasetsLA3IMC/csv/train_ildp.csv -T ../datasetsLA3IMC/csv/test_ildp.csv -r 0.5 -c | grep "Test CCR" | grep -oP "(?<=:).*" | grep "+-" | sed "s/.$//" >> experiments_results.out')

# EXPERIMENT 5

os.system("echo 'NOMNIST DATASET' >> experiments_results.out")

os.system('python3 rbf.py -t ../datasetsLA3IMC/csv/train_nomnist.csv -T ../datasetsLA3IMC/csv/test_nomnist.csv -r 0.05 -c | grep "Test CCR" | grep -oP "(?<=:).*" | grep "+-" | sed "s/.$//" >> experiments_results.out')

# Do the same for r 0.15, 0.25 and 0.5 and train_nomnist.csv

os.system('python3 rbf.py -t ../datasetsLA3IMC/csv/train_nomnist.csv -T ../datasetsLA3IMC/csv/test_nomnist.csv -r 0.15 -c | grep "Test CCR" | grep -oP "(?<=:).*" | grep "+-" | sed "s/.$//" >> experiments_results.out')

os.system('python3 rbf.py -t ../datasetsLA3IMC/csv/train_nomnist.csv -T ../datasetsLA3IMC/csv/test_nomnist.csv -r 0.25 -c | grep "Test CCR" | grep -oP "(?<=:).*" | grep "+-" | sed "s/.$//" >> experiments_results.out')

os.system('python3 rbf.py -t ../datasetsLA3IMC/csv/train_nomnist.csv -T ../datasetsLA3IMC/csv/test_nomnist.csv -r 0.5 -c | grep "Test CCR" | grep -oP "(?<=:).*" | grep "+-" | sed "s/.$//" >> experiments_results.out')

os.system("echo 'END OF EXPERIMENTS 1 =================' >> experiments_results.out")

######################### EXPERIMENT 2 #########################

os.system("echo 'EXPERIMENT 2 (classification, eta and regularization' >> experiments_results.out")

# EXPERIMENT 1

os.system("echo 'SIN DATASET' >> experiments_results.out")

