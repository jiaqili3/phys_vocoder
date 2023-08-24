import glob

paths = []
paths += glob.glob('transfer_attack_exps/mate_*_UNetMixedLoss1*/log')
# sort the paths
paths = sorted(paths)
for path in paths:
    # print the path and the last line in the path file
    with open(path, 'r') as f:
        lines = f.readlines()
        print(path, lines[-1].split()[-1])