import os
import numpy as np
import sys
from alive_progress import alive_bar

print(f">>> {sys.argv[2]} OPTIMIZATION <<<")

train_dataset_path = f'./KaggleDataset/train_{sys.argv[2]}.csv'
test_dataset_path = f'./KaggleDataset/val_{sys.argv[2]}.csv'
experiment_name = f"{sys.argv[2]}_optimized_dataset_v2_{sys.argv[1]}"

execution_mode = "F1"  # CCR it's available too
n_seeds = 5  # Number of seeds used for RBF, this value it's necessary for make the loop for finding the best value
outputs = 1  # Number of outputs
# found

# This new script select the best seed found

# RBF PARAMETERS
learning_rates = np.linspace(10e-10, 1, 10)
c_flag = np.array([True])
ratio_rbf = np.linspace(0.05, 0.5, 10)
regularizations = np.array(['l1', 'l2'])
fairness_flag = np.array([False])
radii_adjustment_heuristic = np.array(
    ['mean_radii_centroids',
     'max_distance_between_centers',
     'mean_distance_between_centers',
     'min_max_distance_between_centers'])  # 'max_distance_between_centers', 'mean_distance_between_centers' can be
# added too

total = len(learning_rates) * len(c_flag) * len(ratio_rbf) * len(regularizations) * len(fairness_flag) * len(
    radii_adjustment_heuristic)

if execution_mode == "MSE":
    best_test_error = sys.float_info.max

    with alive_bar(total) as bar:
        for learning_rate in learning_rates:
            for c in c_flag:
                for ratio in ratio_rbf:
                    for regularization in regularizations:
                        for fairness in fairness_flag:
                            for radii_adjustment in radii_adjustment_heuristic:
                                os.system(
                                    f"python3 rbf_v2.py -t {train_dataset_path} -T {test_dataset_path} -e {learning_rate} {'-c' if c == True else ''} -r {ratio} {'-l' if regularization == 'l2' else ''} {'-f' if fairness == True else ''} -o {outputs} -H {radii_adjustment} | grep \"Test MSE\" | grep -oP '(?<=:).*' | grep -v \"+-\" > .trash_{sys.argv[1]}_{sys.argv[2]}")

                                with open(f".trash_{sys.argv[1]}_{sys.argv[2]}", "r") as f:
                                    lines = f.readlines()

                                    current_best_seed_test_error = float(lines[0])
                                    current_best_seed = 0 + 1

                                    for index in range(1, n_seeds):
                                        if current_best_seed_test_error > float(lines[index]):
                                            current_best_seed_test_error = float(lines[index])
                                            current_best_seed = index + 1

                                    if current_best_seed_test_error < best_test_error:
                                        print(
                                            f"\t>>> 🥇 New best test error: {current_best_seed_test_error} vs {best_test_error}")
                                        best_test_error = current_best_seed_test_error
                                        best_learning_rate = learning_rate
                                        best_c = c
                                        best_ratio = ratio
                                        best_regularization = regularization
                                        best_fairness = fairness
                                        best_seed = current_best_seed
                                        best_radii_adjustment = radii_adjustment

                                        with open(f'{experiment_name}_best_params.out', 'w') as f:
                                            f.write(">>> MSE MODE <<<\n\n")
                                            f.write(f"Best seed: {best_seed}\n")
                                            f.write(f"Best learning rate: {best_learning_rate}\n")
                                            f.write(f"Best c: {best_c}\n")
                                            f.write(f"Best ratio: {best_ratio}\n")
                                            f.write(f"Best regularization: {best_regularization}\n")
                                            f.write(f"Best fairness: {best_fairness}\n")
                                            f.write(f"Best test error: {best_test_error}\n\n")
                                            f.write(f"Best radii adjustment heuristic: {best_radii_adjustment}\n")
                                            f.write(
                                                f"Command for execute the experiment: python3 rbf_v2.py -t {train_dataset_path} -T {test_dataset_path} -e {best_learning_rate} {'-c' if best_c == True else ''} -r {best_ratio} {'-l' if best_regularization == 'l2' else ''} {'-f' if best_fairness == True else ''} -o {outputs} -H {best_radii_adjustment}")

                                    else:
                                        print(
                                            f"\t>>> 🥉 Current test error: {current_best_seed_test_error} vs {best_test_error}")

                                bar()

    with open(f'{experiment_name}_best_params.out', 'w') as f:
        f.write(">>> MSE MODE <<<\n\n")
        f.write(f"Best seed: {best_seed}\n")
        f.write(f"Best learning rate: {best_learning_rate}\n")
        f.write(f"Best c: {best_c}\n")
        f.write(f"Best ratio: {best_ratio}\n")
        f.write(f"Best regularization: {best_regularization}\n")
        f.write(f"Best fairness: {best_fairness}\n")
        f.write(f"Best test error: {best_test_error}\n\n")
        f.write(f"Best radii adjustment heuristic: {best_radii_adjustment}\n")
        f.write(
            f"Command for execute the experiment: python3 rbf_v2.py -t {train_dataset_path} -T {test_dataset_path} -e {best_learning_rate} {'-c' if best_c == True else ''} -r {best_ratio} {'-l' if best_regularization == 'l2' else ''} {'-f' if best_fairness == True else ''} -o {outputs} -H {best_radii_adjustment}")

elif execution_mode == "CCR":  # The same but for CCR
    best_test_ccr = 0

    with alive_bar(total) as bar:
        for learning_rate in learning_rates:
            for c in c_flag:
                for ratio in ratio_rbf:
                    for regularization in regularizations:
                        for fairness in fairness_flag:
                            for radii_adjustment in radii_adjustment_heuristic:
                                os.system(
                                    f"python3 rbf_v2.py -t {train_dataset_path} -T {test_dataset_path} -e {learning_rate} {'-c' if c == True else ''} -r {ratio} {'-l' if regularization == 'l2' else ''} {'-f' if fairness == True else ''} -o {outputs} -H {radii_adjustment} | grep \"Test CCR\" | grep -oP '(?<=:).*' | grep -v \"+-\" | sed 's/.$//' > .trash_{sys.argv[1]}_{sys.argv[2]}")

                                with open(f".trash_{sys.argv[1]}_{sys.argv[2]}", "r") as f:
                                    lines = f.readlines()

                                    current_best_seed_test_ccr = float(lines[0])
                                    current_best_seed = 0 + 1

                                    for index in range(1, n_seeds):
                                        if float(lines[index]) > current_best_seed_test_ccr:
                                            current_best_seed_test_ccr = float(lines[index])
                                            current_best_seed = index + 1

                                    if current_best_seed_test_ccr > best_test_ccr:
                                        print(
                                            f"\t>>> 🥇 New best test CCR: {current_best_seed_test_ccr} vs {best_test_ccr}")
                                        best_test_ccr = current_best_seed_test_ccr
                                        best_learning_rate = learning_rate
                                        best_c = c
                                        best_ratio = ratio
                                        best_regularization = regularization
                                        best_fairness = fairness
                                        best_radii_adjustment = radii_adjustment
                                        best_seed = current_best_seed

                                        with open(f'{experiment_name}_best_params.out', 'w') as f:
                                            f.write(">>> CCR MODE <<<\n\n")
                                            f.write(f"Best seed: {best_seed}\n")
                                            f.write(f"Best learning rate: {best_learning_rate}\n")
                                            f.write(f"Best c: {best_c}\n")
                                            f.write(f"Best ratio: {best_ratio}\n")
                                            f.write(f"Best regularization: {best_regularization}\n")
                                            f.write(f"Best fairness: {best_fairness}\n")
                                            f.write(f"Best test ccr: {best_test_ccr}\n\n")
                                            f.write(f"Best radii adjustment heuristic: {best_radii_adjustment}\n")
                                            f.write(
                                                f"Command for execute the experiment: python3 rbf_2.py -t {train_dataset_path} -T {test_dataset_path} -e {best_learning_rate} {'-c' if best_c == True else ''} -r {best_ratio} {'-l' if best_regularization == 'l2' else ''} {'-f' if best_fairness == True else ''} -o {outputs} -H {best_radii_adjustment}")

                                    else:
                                        print(
                                            f"\t>>> 🥉 Current test CCR: {current_best_seed_test_ccr} vs {best_test_ccr}")

                                bar()

    with open(f'{experiment_name}_best_params.out', 'w') as f:
        f.write(">>> CCR MODE <<<\n\n")
        f.write(f"Best seed: {best_seed}\n")
        f.write(f"Best learning rate: {best_learning_rate}\n")
        f.write(f"Best c: {best_c}\n")
        f.write(f"Best ratio: {best_ratio}\n")
        f.write(f"Best regularization: {best_regularization}\n")
        f.write(f"Best fairness: {best_fairness}\n")
        f.write(f"Best test ccr: {best_test_ccr}\n\n")
        f.write(f"Best radii adjustment heuristic: {best_radii_adjustment}\n")
        f.write(
            f"Command for execute the experiment: python3 rbf_v2.py -t {train_dataset_path} -T {test_dataset_path} -e {best_learning_rate} {'-c' if best_c == True else ''} -r {best_ratio} {'-l' if best_regularization == 'l2' else ''} {'-f' if best_fairness == True else ''} -o {outputs} -H {best_radii_adjustment}")

elif execution_mode == "F1":  # The same but for F1
    best_test_f1 = 0

    with alive_bar(total) as bar:
        for learning_rate in learning_rates:
            for c in c_flag:
                for ratio in ratio_rbf:
                    for regularization in regularizations:
                        for fairness in fairness_flag:
                            for radii_adjustment in radii_adjustment_heuristic:
                                os.system(
                                    f"python3 rbf_v2.py -t {train_dataset_path} -T {test_dataset_path} -e {learning_rate} {'-c' if c == True else ''} -r {ratio} {'-l' if regularization == 'l2' else ''} {'-f' if fairness == True else ''} -o {outputs} -H {radii_adjustment} | grep \"Test F1\" | grep -oP '(?<=:).*' | grep -v \"+-\" | sed 's/.$//' > .trash_{sys.argv[1]}_{sys.argv[2]}")

                                with open(f".trash_{sys.argv[1]}_{sys.argv[2]}", "r") as f:
                                    lines = f.readlines()

                                    current_best_seed_test_f1 = float(lines[0])
                                    current_best_seed = 0 + 1

                                    for index in range(1, n_seeds):
                                        if float(lines[index]) > current_best_seed_test_f1:
                                            current_best_seed_test_f1 = float(lines[index])
                                            current_best_seed = index + 1

                                    if current_best_seed_test_f1 > best_test_f1:
                                        print(
                                            f"\t>>> 🥇 New best test F1: {current_best_seed_test_f1} vs {best_test_f1}")
                                        best_test_f1 = current_best_seed_test_f1
                                        best_learning_rate = learning_rate
                                        best_c = c
                                        best_ratio = ratio
                                        best_regularization = regularization
                                        best_fairness = fairness
                                        best_radii_adjustment = radii_adjustment
                                        best_seed = current_best_seed

                                        with open(f'{experiment_name}_best_params.out', 'w') as f:
                                            f.write(">>> F1 MODE <<<\n\n")
                                            f.write(f"Best seed: {best_seed}\n")
                                            f.write(f"Best learning rate: {best_learning_rate}\n")
                                            f.write(f"Best c: {best_c}\n")
                                            f.write(f"Best ratio: {best_ratio}\n")
                                            f.write(f"Best regularization: {best_regularization}\n")
                                            f.write(f"Best fairness: {best_fairness}\n")
                                            f.write(f"Best test f1: {best_test_f1}\n\n")
                                            f.write(f"Best radii adjustment heuristic: {best_radii_adjustment}\n")
                                            f.write(f"Command for execute the experiment: python3 rbf_v2.py -t {train_dataset_path} -T {test_dataset_path} -e {best_learning_rate} {'-c' if best_c == True else ''} -r {best_ratio} {'-l' if best_regularization == 'l2' else ''} {'-f' if best_fairness == True else ''} -o {outputs} -H {best_radii_adjustment}")

                                    else:
                                        print(
                                            f"\t>>> 🥉 Current test F1: {current_best_seed_test_f1} vs {best_test_f1}")

                                bar()

    with open(f'{experiment_name}_best_params.out', 'w') as f:
        f.write(">>> F1 MODE <<<\n\n")
        f.write(f"Best seed: {best_seed}\n")
        f.write(f"Best learning rate: {best_learning_rate}\n")
        f.write(f"Best c: {best_c}\n")
        f.write(f"Best ratio: {best_ratio}\n")
        f.write(f"Best regularization: {best_regularization}\n")
        f.write(f"Best fairness: {best_fairness}\n")
        f.write(f"Best test f1: {best_test_f1}\n\n")
        f.write(f"Best radii adjustment heuristic: {best_radii_adjustment}\n")
        f.write(f"Command for execute the experiment: python3 rbf_v2.py -t {train_dataset_path} -T {test_dataset_path} -e {best_learning_rate} {'-c' if best_c == True else ''} -r {best_ratio} {'-l' if best_regularization == 'l2' else ''} {'-f' if best_fairness == True else ''} -o {outputs} -H {best_radii_adjustment}")

os.system(f"rm .trash_{sys.argv[1]}_{sys.argv[2]}")
