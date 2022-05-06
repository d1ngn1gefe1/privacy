import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='darkgrid')

plt.plot([2, 4], [50.15, 74.14], label='LP-2ST', marker='o')
plt.plot([1, 2, 4, 8], [82.32, 84.35, 85.79, 87.03], label='Ours', marker='o')

plt.xlabel('Privacy (epsilon)')
plt.ylabel('Utility (accuracy)')

plt.legend()
plt.show()
