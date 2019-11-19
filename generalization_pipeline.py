import subprocess
def exe(string):
    print (string)
    subprocess.call(string, shell=True)



source_types = ['PER', 'W_PER', 'INS', 'W_INS', 'DEL', 'W_DEL', 'SUB', 'W_SUB', 'NOISE_ALL']
target_types = ['PER', 'W_PER', 'INS', 'W_INS', 'DEL', 'W_DEL', 'SUB', 'W_SUB']
for source in source_types:
    for target in target_types:
        if source != target:
            exe('python generalization_experiment.py --source {} --target {}'.format(source, target))
