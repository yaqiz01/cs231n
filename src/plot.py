from os import listdir
from os.path import isfile, isdir, join, splitext, basename, dirname
import numpy as np
import matplotlib

RESULT_PATH='../results'

logs = [f for f in listdir(RESULT_PATH) if isfile(join(RESULT_PATH, f)) and f.endswith('.log')]

# parse logs
results = {}
for log in logs:
    with open(join(RESULT_PATH, log), 'r') as f:
        results[log] = {}
        for line in f:
            if 'Configuration' in line:
                key, val = line.split(' ')[1].split('=')
                results[log][key] = val
            if 'Error' in line or 'Fail' in line: # broken log
                results.pop(log, None)
                continue
