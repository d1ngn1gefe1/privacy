import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')

with sns.color_palette(sns.color_palette('Set2')):
  sns.lineplot([2, 4, 8], [67, 70, 73], label='DP-SGD [17]', marker='s', linestyle='solid')
  sns.lineplot([1, 2, 3], [52.95, 65.03, 68.33], label='DPNAS [54]', marker='X', linestyle='solid')
  sns.lineplot([7.53], [66.2], label='Tempered Sigmoids [51]', marker='D', linestyle='solid')
  sns.lineplot([3], [69.3], label='DP-ScatterNet [19]', marker='D')
  sns.lineplot([2, 4, 8], [77.4, 79.1, 79.5], label='Norm-DP-SGD [50]', marker='o', linestyle='solid')
  sns.lineplot([1.93, 4.21, 7.42], [58.6, 66.2, 70.1], label='Dormann et al. [46]', marker='o', linestyle='solid')
  sns.lineplot([1.99, 5.01, 7.01, 10.00], [50.85, 61.75, 62.32, 64.73], label='DPlis [45]', marker='s', linestyle='solid')
  sns.lineplot([0.5, 1.0, 1.5], [73.28, 76.64, 81.57], label='Scalable [56]', marker='X', linestyle='solid')
  sns.lineplot([2.00], [92.70], label='DP-ScatterNet [19]', marker='D')


xs = [1, 2, 4, 8]
ys = [96.03, 96.07, 96.29, 96.54]
sns.lineplot(xs, ys, label='Ours (ViT-S/16-224)', marker='s', color='darkolivegreen', linestyle='solid')
for x, y in zip(xs, ys):
  plt.text(x-0.3, y+1, str(y), color='darkolivegreen', fontsize=12)

axes = plt.gca()
axes.xaxis.grid()
axes.yaxis.grid()
plt.xlabel('Privacy (epsilon)')
plt.ylabel('Utility (accuracy)')

plt.xlim(0, 10)
plt.ylim(40, 100)
plt.legend(loc='lower right', ncol=2)
plt.tight_layout()
plt.show()
