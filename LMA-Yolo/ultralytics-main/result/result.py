import pandas as pd
import matplotlib.pyplot as plt

# 读取两个CSV文件
file_path1 = r'/root/autodl-tmp/ultralytics-main/result/results.csv'  # Baseline的CSV文件路径
file_path2 = r'/root/autodl-tmp/ultralytics-main/result/sd_all.csv'  # Ours的CSV文件路径

data1 = pd.read_csv(file_path1)
data2 = pd.read_csv(file_path2)

# 去除列名中的空格
data1.columns = data1.columns.str.strip()
data2.columns = data2.columns.str.strip()

# 定义保存图片的函数，确保300 DPI
def save_plot(fig, filename):
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {filename}")

# 设置图表的格式（符合IEEE标准）
plt.rcParams.update({
    'font.size': 12,               # 字体大小
    'axes.titlesize': 14,          # 标题字体大小
    'axes.labelsize': 12,          # 坐标轴标签字体大小
    'legend.fontsize': 12,         # 图例字体大小
    'xtick.labelsize': 10,         # x轴刻度字体大小
    'ytick.labelsize': 10,         # y轴刻度字体大小
})

# 绘制损失函数对比曲线
if 'epoch' in data1.columns:
    plt.figure(figsize=(10, 6))
    plt.plot(data1['epoch'], data1['train/box_loss'], label='Training Box Loss - Baseline', linewidth=1.5)
    plt.plot(data2['epoch'], data2['train/box_loss'], label='Training Box Loss - Ours', linestyle='--', linewidth=1.5)
    plt.plot(data1['epoch'], data1['val/box_loss'], label='Validation Box Loss - Baseline', linewidth=1.5)
    plt.plot(data2['epoch'], data2['val/box_loss'], label='Validation Box Loss - Ours', linestyle='--', linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Box Loss')
    plt.title('Box Loss Comparison')
    legend = plt.legend(loc='best', frameon=True, framealpha=1, edgecolor='black')
    legend.get_frame().set_linewidth(1.0)
    plt.grid(color='gray', linestyle=':', linewidth=0.5)
    save_plot(plt.gcf(), 'box_loss_comparison.png')
    plt.show()

# 绘制学习率曲线
if 'epoch' in data1.columns and 'lr/pg0' in data1.columns:
    plt.figure(figsize=(10, 6))
    plt.plot(data1['epoch'], data1['lr/pg0'], label='Learning Rate - Baseline', linewidth=1.5)
    plt.plot(data2['epoch'], data2['lr/pg0'], label='Learning Rate - Ours', linestyle='--', linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Comparison')
    legend = plt.legend(loc='best', frameon=True, framealpha=1, edgecolor='black')
    legend.get_frame().set_linewidth(1.0)
    plt.grid(color='gray', linestyle=':', linewidth=0.5)
    save_plot(plt.gcf(), 'learning_rate_comparison.png')
    plt.show()

# 绘制mAP对比曲线，将mAP@50和mAP@50-95放在同一张图上
if 'epoch' in data1.columns and 'metrics/mAP50-95(B)' in data1.columns and 'metrics/mAP50(B)' in data1.columns:
    plt.figure(figsize=(10, 6))
    # 将mAP值转换为百分比显示
    plt.plot(data1['epoch'], data1['metrics/mAP50(B)'] * 100, label='mAP@50 - Baseline', linewidth=1.5)
    plt.plot(data2['epoch'], data2['metrics/mAP50(B)'] * 100, label='mAP@50 - Ours', linestyle='--', linewidth=1.5)
    plt.plot(data1['epoch'], data1['metrics/mAP50-95(B)'] * 100, label='mAP@50-95 - Baseline', linewidth=1.5)
    plt.plot(data2['epoch'], data2['metrics/mAP50-95(B)'] * 100, label='mAP@50-95 - Ours', linestyle='--', linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('mAP (%)')  # 将纵坐标标签改为百分比
    plt.title('mAP Comparison (mAP@50 and mAP@50-95)')
    plt.yticks(range(0, 101, 10), [f'{i}%' for i in range(0, 101, 10)])  # 设置百分比刻度
    legend = plt.legend(loc='best', frameon=True, framealpha=1, edgecolor='black')
    legend.get_frame().set_linewidth(1.0)
    plt.grid(color='gray', linestyle=':', linewidth=0.5)
    save_plot(plt.gcf(), 'mAP_comparison.png')
    plt.show()

# 绘制PR曲线（Precision-Recall Curve）
if 'metrics/precision(B)' in data1.columns and 'metrics/recall(B)' in data1.columns:
    plt.figure(figsize=(10, 6))
    plt.plot(data1['metrics/recall(B)'], data1['metrics/precision(B)'], label='PR Curve - Baseline', linewidth=1.5)
    plt.plot(data2['metrics/recall(B)'], data2['metrics/precision(B)'], label='PR Curve - Ours', linestyle='--', linewidth=1.5)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    legend = plt.legend(loc='best', frameon=True, framealpha=1, edgecolor='black')
    legend.get_frame().set_linewidth(1.0)
    plt.grid(color='gray', linestyle=':', linewidth=0.5)
    save_plot(plt.gcf(), 'pr_curve_comparison.png')
    plt.show()

# 绘制Recall和Precision随着Epoch变化的对比曲线
if 'epoch' in data1.columns and 'metrics/precision(B)' in data1.columns and 'metrics/recall(B)' in data1.columns:
    plt.figure(figsize=(10, 6))
    
    # 绘制Recall曲线
    plt.plot(data1['epoch'], data1['metrics/recall(B)'] * 100, label='Recall - Baseline', linewidth=1.5)
    plt.plot(data2['epoch'], data2['metrics/recall(B)'] * 100, label='Recall - Ours', linestyle='--', linewidth=1.5)
    
    # 绘制Precision曲线
    plt.plot(data1['epoch'], data1['metrics/precision(B)'] * 100, label='Precision - Baseline', linewidth=1.5)
    plt.plot(data2['epoch'], data2['metrics/precision(B)'] * 100, label='Precision - Ours', linestyle='--', linewidth=1.5)
    
    plt.xlabel('Epoch')
    plt.ylabel('Percentage (%)')  # 将纵坐标标签改为百分比
    plt.title('Recall and Precision Comparison over Epochs')
    
    # 设置百分比刻度
    plt.yticks(range(0, 101, 10), [f'{i}%' for i in range(0, 101, 10)])
    
    legend = plt.legend(loc='best', frameon=True, framealpha=1, edgecolor='black')
    legend.get_frame().set_linewidth(1.0)
    
    plt.grid(color='gray', linestyle=':', linewidth=0.5)
    
    save_plot(plt.gcf(), 'recall_precision_epoch_comparison.png')
    plt.show()
