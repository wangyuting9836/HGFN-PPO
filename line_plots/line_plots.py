import matplotlib.pyplot as plt
import json

with open('training process.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

fig, axes = plt.subplots(3, 1, figsize=(12, 15))
fig.suptitle('Training Process Metrics', fontsize=16, fontweight='bold')

axes = axes.flatten()

indexes = [1, 2, 9]
max_x = [1200, 1200, 120]
for i, index in enumerate(indexes):
    window_key = f"window{index}"
    window_data = data["jsons"][window_key]

    # 提取x和y数据
    x_data = window_data["content"]["data"][0]["x"]
    y_data = window_data["content"]["data"][0]["y"]
    title = window_data["title"]

    axes[i].plot(x_data[0:max_x[i]], y_data[0:max_x[i]], 'b-', linewidth=2, marker=None, markersize=4)
    axes[i].set_title(title, fontsize=14, fontweight='bold')
    axes[i].set_xlabel('episode')
    axes[i].set_ylabel('value')
    axes[i].grid(True, alpha=0.3)

    if window_key == "window1":  # true return
        axes[i].set_title('return')
        axes[i].set_ylabel('return')
    elif window_key == "window6":  # explained variance
        axes[i].set_ylabel('explained variance')
    elif window_key == "window9":  # make span
        axes[i].set_ylabel('makespan')

plt.tight_layout()
plt.subplots_adjust(top=0.95)

# plt.show()
plt.savefig('training_process_metrics.svg')

