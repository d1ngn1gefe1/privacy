import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')

xs = [1, 2, 4, 8]
ys1 = [74.18, 77.01, 78.97, 80.86]
ys2 = [77.45, 78.32, 79.99, 82.81]
sns.lineplot(xs, ys1, label='$\epsilon$ = 5', marker='o', linestyle='solid', color='tomato')
sns.lineplot(xs, ys2, label='$\epsilon$ = 10', marker='o', linestyle='solid', color='cadetblue')
for x, y in zip(xs, ys1):
    plt.text(x-0.3, y+0.1, str(y), color='tomato', fontsize=12)
for x, y in zip(xs, ys2):
    plt.text(x-0.3, y+0.1, str(y), color='cadetblue', fontsize=12)

axes = plt.gca()
axes.xaxis.grid()
axes.yaxis.grid()
plt.xlabel('Number of clips')
plt.ylabel('Top-1 accuracy')

plt.legend()
plt.tight_layout()
plt.show()
