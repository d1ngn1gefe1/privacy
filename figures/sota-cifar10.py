import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='darkgrid')

plt.plot([2, 4, 8], [67, 70, 73], label='DP-SGD', marker='o')
plt.plot([1, 2, 3], [52.95, 65.03, 68.33], label='DPNAS', marker='o')
plt.plot([7.53], [66.2], label='Tempered Sigmoids', marker='o')
plt.plot([3], [69.3], label='ScatterNet', marker='o')
plt.plot([2, 4, 8], [77.4, 79.1, 79.5], label='Norm-DP-SGD', marker='o')
plt.plot([1.93, 4.21, 7.42], [58.6, 66.2, 70.1], label='Dormann et al.', marker='o')

plt.xlabel('Privacy (epsilon)')
plt.ylabel('Utility (accuracy)')

plt.legend()
plt.show()
