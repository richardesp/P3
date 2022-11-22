# BEST RATIO RBF = 0.05

import os

os.system("echo 'ILDP DATASET' >> experiments_results.out")

os.system('python3 rbf.py -t ../datasetsLA3IMC/csv/train_ildp.csv -T ../datasetsLA3IMC/csv/test_ildp.csv -r 0.05 -c | grep "Test CCR" | grep -oP "(?<=:).*" | grep "+-" | sed "s/.$//" >> experiments_results.out')

# Do the same for r 0.15, 0.25 and 0.5 and train_ildp.csv

os.system('python3 rbf.py -t ../datasetsLA3IMC/csv/train_ildp.csv -T ../datasetsLA3IMC/csv/test_ildp.csv -r 0.05 -e 1 -c | grep "Test CCR" | grep -oP "(?<=:).*" | grep "+-" | sed "s/.$//" >> experiments_results.out')

os.system('python3 rbf.py -t ../datasetsLA3IMC/csv/train_ildp.csv -T ../datasetsLA3IMC/csv/test_ildp.csv -r 0.05 -e 0.1 -c | grep "Test CCR" | grep -oP "(?<=:).*" | grep "+-" | sed "s/.$//" >> experiments_results.out')

os.system('python3 rbf.py -t ../datasetsLA3IMC/csv/train_ildp.csv -T ../datasetsLA3IMC/csv/test_ildp.csv -r 0.05 -e 0.01 -c | grep "Test CCR" | grep -oP "(?<=:).*" | grep "+-" | sed "s/.$//" >> experiments_results.out')

os.system('python3 rbf.py -t ../datasetsLA3IMC/csv/train_ildp.csv -T ../datasetsLA3IMC/csv/test_ildp.csv -r 0.05 -e 0.001 -c | grep "Test CCR" | grep -oP "(?<=:).*" | grep "+-" | sed "s/.$//" >> experiments_results.out')

os.system('python3 rbf.py -t ../datasetsLA3IMC/csv/train_ildp.csv -T ../datasetsLA3IMC/csv/test_ildp.csv -r 0.05 -e 0.0000000001 -c | grep "Test CCR" | grep -oP "(?<=:).*" | grep "+-" | sed "s/.$//" >> experiments_results.out')

os.system('python3 rbf.py -t ../datasetsLA3IMC/csv/train_ildp.csv -T ../datasetsLA3IMC/csv/test_ildp.csv -r 0.05 -e 1 --l2 -c | grep "Test CCR" | grep -oP "(?<=:).*" | grep "+-" | sed "s/.$//" >> experiments_results.out')

os.system('python3 rbf.py -t ../datasetsLA3IMC/csv/train_ildp.csv -T ../datasetsLA3IMC/csv/test_ildp.csv -r 0.05 -e 0.1 --l2 -c | grep "Test CCR" | grep -oP "(?<=:).*" | grep "+-" | sed "s/.$//" >> experiments_results.out')

os.system('python3 rbf.py -t ../datasetsLA3IMC/csv/train_ildp.csv -T ../datasetsLA3IMC/csv/test_ildp.csv -r 0.05 -e 0.01 --l2 -c | grep "Test CCR" | grep -oP "(?<=:).*" | grep "+-" | sed "s/.$//" >> experiments_results.out')

os.system('python3 rbf.py -t ../datasetsLA3IMC/csv/train_ildp.csv -T ../datasetsLA3IMC/csv/test_ildp.csv -r 0.05 -e 0.001 --l2 -c | grep "Test CCR" | grep -oP "(?<=:).*" | grep "+-" | sed "s/.$//" >> experiments_results.out')

os.system('python3 rbf.py -t ../datasetsLA3IMC/csv/train_ildp.csv -T ../datasetsLA3IMC/csv/test_ildp.csv -r 0.05 -e 0.0000000001 --l2 -c | grep "Test CCR" | grep -oP "(?<=:).*" | grep "+-" | sed "s/.$//" >> experiments_results.out')


