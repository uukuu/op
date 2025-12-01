import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import pandas as pd
import seaborn as sns

# ==========================================
# 1. 配置与模型定义 (必须与训练代码一致)
# ==========================================

# 设置绘图风格，适合学术报告
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif' 
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.unicode_minus'] = False 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 模型文件存放目录 (请根据实际情况修改)
MODEL_DIR = "models" 
# 图像输出目录
OUTPUT_DIR = "analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class FCNet(nn.Module):
    """需与 main.py 中的定义完全一致"""
    def __init__(self, in_dim=2, out_dim=1, hidden_dim=20, n_hidden=3):
        super(FCNet, self).__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        X = torch.cat([x, y], dim=1)
        return self.net(X)

def exact_u(x, y):
    """精确解：用于计算误差"""
    return np.sin(np.pi * x) * np.sin(np.pi * y)

# ==========================================
# 2. 核心分析功能
# ==========================================

def load_and_evaluate(model_path, resolution=256):
    """加载指定路径的模型并计算 L2 误差"""
    if not os.path.exists(model_path):
        return None, None, None, None, None

    # 初始化模型结构
    model = FCNet(hidden_dim=20, n_hidden=3).to(device)
    
    # 加载权重
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None, None, None, None, None

    model.eval()

    # 生成高分辨率测试网格
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    x_flat = torch.tensor(X.reshape(-1, 1), dtype=torch.float32).to(device)
    y_flat = torch.tensor(Y.reshape(-1, 1), dtype=torch.float32).to(device)

    # 预测
    with torch.no_grad():
        u_pred = model(x_flat, y_flat).cpu().numpy().reshape(X.shape)
    
    # 计算精确解与误差
    u_true = exact_u(X, Y)
    abs_error = np.abs(u_pred - u_true)
    
    # 计算相对 L2 误差
    l2_error = np.linalg.norm(abs_error) / np.linalg.norm(u_true)
    
    return l2_error, X, Y, u_pred, abs_error

def parse_filename(filename):
    """解析文件名: Adam-400-100-0p1.pt -> (Adam, 400, 100, 0.1)"""
    try:
        name, _ = os.path.splitext(filename)
        parts = name.split('-')
        opt = parts[0]
        N = int(parts[1])
        M = int(parts[2])
        lam_str = parts[3].replace('p', '.')
        lam = float(lam_str)
        return opt, N, M, lam
    except:
        return None

# ==========================================
# 3. 批量处理与绘图
# ==========================================

def batch_analysis():
    print("开始批量分析...")
    data = []
    
    # 1. 遍历文件夹读取所有模型
    if not os.path.exists(MODEL_DIR):
        print(f"错误：文件夹 {MODEL_DIR} 不存在。请创建该文件夹并将 .pt 文件放入其中。")
        return

    files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pt')]
    print(f"找到 {len(files)} 个模型文件。")
    
    best_model_info = {'error': float('inf'), 'data': None, 'name': ''}

    for i, f in enumerate(files):
        params = parse_filename(f)
        if params is None:
            continue
            
        opt, N, M, lam = params
        path = os.path.join(MODEL_DIR, f)
        
        error, X, Y, pred, err_map = load_and_evaluate(path)
        
        if error is not None:
            data.append({
                "Optimizer": opt,
                "N": N,
                "M": M,
                "Lambda": lam,
                "L2_Error": error,
                "Log_Error": np.log10(error)
            })
            
            # 记录最佳模型用于稍后画 3D 图
            if error < best_model_info['error']:
                best_model_info['error'] = error
                best_model_info['data'] = (X, Y, pred, err_map)
                best_model_info['name'] = f"{opt} (N={N}, M={M}, $\lambda$={lam})"

        if (i+1) % 10 == 0:
            print(f"已处理 {i+1}/{len(files)} 个文件...")

    if not data:
        print("没有成功加载任何模型数据。")
        return

    df = pd.DataFrame(data)
    csv_path = os.path.join(OUTPUT_DIR, "experiment_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"数据汇总已保存至: {csv_path}")

    # ===========================
    # 绘图 1: 参数灵敏度热力图 (N vs M)
    # ===========================
    # 为了画 N vs M，我们需要固定 Optimizer 和 Lambda (取平均或取特定值)
    # 这里展示 Adam 优化器下，Lambda=1.0 时的情况
    target_opt = "Adam"
    target_lam = 1.0
    subset = df[(df["Optimizer"] == target_opt) & (df["Lambda"] == target_lam)]
    
    if not subset.empty:
        pivot = subset.pivot(index="N", columns="M", values="Log_Error")
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis_r", cbar_kws={'label': r'$\log_{10}(L^2 \text{ Error})$'})
        plt.title(f"Grid Sensitivity: N vs M ({target_opt}, $\lambda={target_lam}$)")
        plt.xlabel("Boundary Points (M)")
        plt.ylabel("Interior Points (N)")
        plt.savefig(os.path.join(OUTPUT_DIR, "heatmap_N_vs_M.png"), dpi=300)
        plt.close()
        print("已生成：heatmap_N_vs_M.png")

    # ===========================
    # 绘图 2: 惩罚系数 Lambda 的影响 (折线图)
    # ===========================
    # 固定 M=400, 展示不同 N 下 Lambda 对误差的影响
    target_M = 400
    subset_lam = df[(df["Optimizer"] == target_opt) & (df["M"] == target_M)]
    
    if not subset_lam.empty:
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=subset_lam, x="Lambda", y="L2_Error", hue="N", style="N", markers=True, dashes=False, palette="tab10")
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.title(f"Sensitivity of Penalty Coefficient $\lambda$ (M={target_M})")
        plt.ylabel("Relative L2 Error")
        plt.xlabel(r"$\lambda$")
        plt.savefig(os.path.join(OUTPUT_DIR, "sensitivity_lambda.png"), dpi=300)
        plt.close()
        print("已生成：sensitivity_lambda.png")

    # ===========================
    # 绘图 3: 最佳模型 3D 可视化
    # ===========================
    if best_model_info['data']:
        X, Y, pred, err_map = best_model_info['data']
        u_ex = exact_u(X, Y)
        
        fig = plt.figure(figsize=(16, 5))
        
        # 精确解
        ax1 = fig.add_subplot(131, projection='3d')
        surf1 = ax1.plot_surface(X, Y, u_ex, cmap='viridis', antialiased=False, linewidth=0)
        ax1.set_title("Exact Solution")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
        
        # 预测解
        ax2 = fig.add_subplot(132, projection='3d')
        surf2 = ax2.plot_surface(X, Y, pred, cmap='viridis', antialiased=False, linewidth=0)
        ax2.set_title(f"Prediction\n{best_model_info['name']}")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
        
        # 误差
        ax3 = fig.add_subplot(133, projection='3d')
        surf3 = ax3.plot_surface(X, Y, err_map, cmap='inferno', antialiased=False, linewidth=0)
        ax3.set_title("Absolute Error")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "best_solution_3d.png"), dpi=300)
        plt.close()
        print("已生成：best_solution_3d.png")

    # ===========================
    # 绘图 4: 优化器对比 (如果存在多种优化器文件)
    # ===========================
    # 找出每个优化器的最佳表现 (取所有参数组合中的最小值)
    best_per_opt = df.groupby("Optimizer")["L2_Error"].min().reset_index()
    
    if len(best_per_opt) > 1:
        plt.figure(figsize=(7, 5))
        sns.barplot(data=best_per_opt, x="Optimizer", y="L2_Error", palette="viridis")
        plt.yscale("log")
        plt.title("Best Performance Comparison by Optimizer")
        plt.ylabel("Minimum L2 Error Achieved")
        
        # 在柱子上标记数值
        for index, row in best_per_opt.iterrows():
            plt.text(index, row.L2_Error, f"{row.L2_Error:.2e}", color='black', ha="center", va="bottom")
            
        plt.savefig(os.path.join(OUTPUT_DIR, "optimizer_comparison.png"), dpi=300)
        plt.close()
        print("已生成：optimizer_comparison.png")
    else:
        print("仅检测到一种优化器，跳过优化器对比图。")

if __name__ == "__main__":
    batch_analysis()