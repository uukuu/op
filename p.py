import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def draw_flowchart():
    # 创建画布
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 15) # 稍微加宽一点x轴，给右侧连线留空间
    ax.set_ylim(0, 10)
    ax.axis('off') 

    # 定义样式
    box_style = dict(boxstyle="round,pad=0.5", fc="#e6f3ff", ec="#0052cc", lw=2)
    loss_style = dict(boxstyle="round,pad=0.5", fc="#fff2cc", ec="#d6b656", lw=2)
    opt_style = dict(boxstyle="round,pad=0.5", fc="#e2f0d9", ec="#548235", lw=2)
    
    # ====================
    # 1. 绘制节点 (位置保持不变)
    # ====================
    
    # 输入层
    ax.text(1, 5, "输入坐标\n$(x, y)$", ha="center", va="center", size=14, bbox=box_style)
    
    # 神经网络
    ax.text(4, 5, "全连接神经网络\n(FCNN)\n参数 $\\theta$", ha="center", va="center", size=14, bbox=box_style)
    
    # 输出层
    ax.text(7, 5, "预测解\n$\\hat{u}(x, y; \\theta)$", ha="center", va="center", size=14, bbox=box_style)
    
    # 分支1：边界条件
    ax.text(10, 7, "边界点 $(x_b, y_b)$\n计算边界误差", ha="center", va="center", size=12, bbox=dict(boxstyle="square,pad=0.4", fc="white", ec="gray", lw=1, linestyle="--"))
    
    # 分支2：自动微分
    ax.text(10, 3, "自动微分 (AD)\n$\\frac{\\partial^2 \\hat{u}}{\\partial x^2}, \\frac{\\partial^2 \\hat{u}}{\\partial y^2}$", ha="center", va="center", size=12, bbox=dict(boxstyle="square,pad=0.4", fc="white", ec="gray", lw=1, linestyle="--"))
    
    # Loss 项
    ax.text(12, 7, "边界损失\n$\\mathcal{L}_{BC}$", ha="center", va="center", size=14, bbox=loss_style)
    ax.text(12, 3, "物理残差损失\n$\\mathcal{L}_{PDE}$", ha="center", va="center", size=14, bbox=loss_style)
    
    # 总 Loss
    ax.text(12, 5, "加权总损失\n$J = \\mathcal{L}_{PDE} + \\lambda \\mathcal{L}_{BC}$", ha="center", va="center", size=14, bbox=loss_style)
    
    # 优化器
    ax.text(8, 1, "优化器 (Adam / L-BFGS)\n更新参数 $\\theta \\leftarrow \\theta - \\eta \\nabla J$", ha="center", va="center", size=14, bbox=opt_style)

    # ====================
    # 2. 绘制黑色数据流箭头
    # ====================
    
    arrow_props = dict(facecolor='black', edgecolor='black', width=1.5, headwidth=8, headlength=10, shrink=0.05)
    
    ax.annotate("", xy=(2.7, 5), xytext=(1.6, 5), arrowprops=arrow_props) # 输入 -> NN
    ax.annotate("", xy=(6.1, 5), xytext=(5.3, 5), arrowprops=arrow_props) # NN -> 输出
    
    ax.annotate("", xy=(9, 7), xytext=(7.8, 5.5), arrowprops=arrow_props) # 输出 -> 边界
    ax.annotate("", xy=(9, 3), xytext=(7.8, 4.5), arrowprops=arrow_props) # 输出 -> AD
    
    ax.annotate("", xy=(11.2, 7), xytext=(11, 7), arrowprops=arrow_props) # -> BC Loss
    ax.annotate("", xy=(11.1, 3), xytext=(10.9, 3), arrowprops=arrow_props) # -> PDE Loss
    
    ax.annotate("", xy=(12, 5.8), xytext=(12, 6.3), arrowprops=arrow_props) # BC -> Total
    ax.annotate("", xy=(12, 4.2), xytext=(12, 3.7), arrowprops=arrow_props) # PDE -> Total
    
    # ====================
    # 3. 绘制红色反向传播箭头 (U型直角线)
    # ====================
    
    # 定义 U 型路径的关键点
    # 起点: 总Loss右侧 (13.6, 5)
    # 拐点1: 向右延伸到 (14.2, 5)
    # 拐点2: 向下延伸到 (14.2, 1)
    # 终点: 向左回到优化器右侧 (10.2, 1)
    verts = [
        (13.6, 5.0), 
        (14.5, 5.0), 
        (14.5, 1.0),
        (10.2, 1.0),
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
    ]
    path = Path(verts, codes)
    
    # 创建红色箭头
    patch = patches.FancyArrowPatch(
        path=path, 
        arrowstyle='-|>', 
        mutation_scale=20, 
        color='#d9534f', 
        lw=2.5,
        joinstyle='round' # 圆角连接更美观
    )
    ax.add_patch(patch)

    # 补充“反向传播”文字说明 (放在右侧竖线旁边)
    ax.text(14.6, 3.0, "反向传播\n(梯度更新)", color="#d9534f", fontsize=12, ha="left", va="center")
    
    # 补充一条从优化器回到神经网络的箭头
    ax.annotate("", xy=(4, 3.8), xytext=(5.7, 1), arrowprops=dict(facecolor='#d9534f', edgecolor='#d9534f', width=1.5, headwidth=8, headlength=10, connectionstyle="arc3,rad=-0.2"))

    # ====================
    # 4. 标题和保存
    # ====================
    plt.title("物理信息神经网络 (PINN) 求解泊松方程流程图", fontsize=18, y=0.98)
    
    save_path = 'pinn_flowchart.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"流程图已生成并保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    draw_flowchart()