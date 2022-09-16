import os
# Plot experiment results stored in specified directory.
logdir = './data/exp3_8ubs'

xaxis = 'Epoch'
# xaxis = 'Time'
# xaxis = 'TotalEnvInteracts'

value = 'AverageEpRet'
# value = 'FairIdx'
# value = 'TotalThroughput'
# value = 'Collision'
# value = 'MaxEpRet'
# value = 'AverageTestEpRet'
# value = 'AverageQVals'
# value = 'LossQ'

plot_cmd = f'python ./utils/plot.py {logdir} --xaxis={xaxis} --value={value}'
os.system(plot_cmd)
