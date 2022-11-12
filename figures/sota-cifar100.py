import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')

with sns.color_palette('Set2'):
  sns.lineplot([3, 4, 5, 6, 8], [20.96, 46.28, 61.38, 68.34, 73.59], label='LP-1ST [47]', marker='o', linestyle='solid')
  sns.lineplot([3, 4, 5, 6, 8], [28.74, 50.15, 63.51, 70.58, 74.14], label='LP-2ST [47]', marker='o', linestyle='solid')
  sns.lineplot([0.39, 0.54, 0.66, 0.77, 0.86], [35.98, 43.76, 48.00, 51.04, 52.04], label='DPlis [45]', marker='o', linestyle='solid')

xs = [1, 2, 4, 8]
ys = [82.32, 84.35, 85.79, 87.03]
plt.plot(xs, ys, label='Ours (ViT-S/16-224)', marker='o', color='darkolivegreen', linestyle='solid')
for x, y in zip(xs, ys):
  plt.text(x-0.3, y-5, str(y), color='darkolivegreen', fontsize=12)

axes = plt.gca()
axes.xaxis.grid()
axes.yaxis.grid()
plt.xlabel('Privacy (epsilon)')
plt.ylabel('Utility (accuracy)')

plt.legend()
plt.tight_layout()
plt.show()
