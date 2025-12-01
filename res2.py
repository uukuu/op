import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. 基础配置
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')

# --- 中文字体设置核心部分 ---
# 尝试设置中文字体，优先使用 SimHei (Windows), 备选 Heiti (Mac), 再次 DejaVu Sans (Linux)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
plt.rcParams['mathtext.fontset'] = 'cm'     # 数学公式字体保持标准样式

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODEL_DIR = "models" 
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
    return np.sin(np.pi * x) * np.sin(np.pi * y)

# ==========================================
# 2. 核心工具函数
# ==========================================

def load_and_evaluate(model_path, resolution=256):
    """加载模型并返回绘图所需的高分辨率数据"""
    if not os.path.exists(model_path):
        return None

    model = FCNet(hidden_dim=20, n_hidden=3).to(device)
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None

    model.eval()
    # 生成网格
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    x_flat = torch.tensor(X.reshape(-1, 1), dtype=torch.float32).to(device)
    y_flat = torch.tensor(Y.reshape(-1, 1), dtype=torch.float32).to(device)

    with torch.no_grad():
        u_pred = model(x_flat, y_flat).cpu().numpy().reshape(X.shape)
    
    u_true = exact_u(X, Y)
    abs_error = np.abs(u_pred - u_true)
    l2_error = np.linalg.norm(abs_error) / np.linalg.norm(u_true)
    
    return {
        "l2_error": l2_error,
        "X": X,
        "Y": Y,
        "pred": u_pred,
        "exact": u_true,
        "error": abs_error
    }

def parse_filename(filename):
    """解析文件名: Optimizer-N-M-Lambda.pt"""
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
# 3. 绘图函数封装 (中文版)
# ==========================================

def plot_solution_comparison(data_dict, title_info, save_name):
    """绘制 精确解 vs 预测解 vs 误差 的 3D 对比图"""
    X, Y = data_dict["X"], data_dict["Y"]
    u_exact = data_dict["exact"]
    u_pred = data_dict["pred"]
    u_error = data_dict["error"]
    
    fig = plt.figure(figsize=(18, 6)) 
    
    # 中文标题
    fig.suptitle(f"解的对比分析: {title_info}", fontsize=16, y=0.95)

    # 子图1: 精确解
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(X, Y, u_exact, cmap='viridis', antialiased=False, linewidth=0)
    ax1.set_title("精确解 $u(x,y)$", fontsize=12, pad=10)
    ax1.set_xlabel("$x$", fontsize=10)
    ax1.set_ylabel("$y$", fontsize=10)
    fig.colorbar(surf1, ax=ax1, shrink=0.4, aspect=12, pad=0.1)

    # 子图2: 预测解
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(X, Y, u_pred, cmap='viridis', antialiased=False, linewidth=0)
    ax2.set_title("预测解 $\hat{u}(x,y)$", fontsize=12, pad=10, color='blue')
    ax2.set_xlabel("$x$", fontsize=10)
    ax2.set_ylabel("$y$", fontsize=10)
    fig.colorbar(surf2, ax=ax2, shrink=0.4, aspect=12, pad=0.1)

    # 子图3: 绝对误差
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(X, Y, u_error, cmap='inferno', antialiased=False, linewidth=0)
    ax3.set_title(f"绝对误差 $|u - \hat{{u}}|$\n($L^2$ 误差: {data_dict['l2_error']:.2e})", fontsize=12, pad=10, color='red')
    ax3.set_xlabel("$x$", fontsize=10)
    ax3.set_ylabel("$y$", fontsize=10)
    fig.colorbar(surf3, ax=ax3, shrink=0.4, aspect=12, pad=0.1)

    plt.subplots_adjust(left=0.05, right=0.95, wspace=0.3, top=0.85, bottom=0.05)
    
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> 已保存对比图: {save_name}")

def plot_3d_bar_chart(df, optimizer_name):
    """绘制 Ratio vs Lambda 的 3D 柱状图"""
    df = df.copy()
    df['Ratio'] = df['N'] / df['M']
    subset = df[df["Optimizer"] == optimizer_name].copy()
    
    if subset.empty: return

    best_ratio_df = subset.groupby(['Ratio', 'Lambda'])['L2_Error'].min().reset_index()
    # 使用 -log10(Error) 作为高度，越高越好
    best_ratio_df['Score'] = -np.log10(best_ratio_df['L2_Error'] + 1e-16)
    
    ratios = sorted(best_ratio_df['Ratio'].unique())
    lambdas = sorted(best_ratio_df['Lambda'].unique())
    
    x_indices = np.arange(len(lambdas))
    y_indices = np.arange(len(ratios))
    xx, yy = np.meshgrid(x_indices, y_indices)
    
    x = xx.flatten()
    y = yy.flatten()
    z = np.zeros_like(x)
    dx = 0.5 * np.ones_like(x)
    dy = 0.5 * np.ones_like(y)
    
    dz = []
    colors = []
    cmap = plt.cm.viridis
    norm = plt.Normalize(0, len(lambdas)-1)
    
    for y_idx, r in enumerate(ratios):
        for x_idx, l in enumerate(lambdas):
            val = best_ratio_df[(best_ratio_df['Ratio'] == r) & (best_ratio_df['Lambda'] == l)]['Score'].values
            dz.append(val[0] if len(val) > 0 else 0)
            colors.append(cmap(norm(x_idx)))
            
    dz = np.array(dz)
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.bar3d(x - dx/2, y - dy/2, z, dx, dy, dz, color=colors, alpha=0.85, shade=True)
    
    ax.set_xticks(x_indices)
    ax.set_xticklabels([str(l) for l in lambdas], fontsize=10)
    ax.set_xlabel(r'惩罚系数 $\lambda$', fontsize=12, labelpad=15)
    
    ax.set_yticks(y_indices)
    ax.set_yticklabels([f"{r:.1f}" for r in ratios], fontsize=10)
    ax.set_ylabel(r'内部/边界点数比 ($N/M$)', fontsize=12, labelpad=15)
    
    ax.set_zlabel(r'精度评分 ($-\log_{10} L^2$)', fontsize=12, labelpad=15)
    
    ax.set_title(f"优化器性能全景: {optimizer_name}\n(数值越高代表精度越高)", fontsize=16, y=0.95)
    
    ax.view_init(elev=25, azim=-55)
    
    plt.subplots_adjust(left=0, right=1, bottom=0, top=0.9)
    
    save_path = os.path.join(OUTPUT_DIR, f"3d_bar_ratio_lambda_{optimizer_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> 已保存 3D 柱状图: {save_path}")

# ==========================================
# 4. 主程序
# ==========================================

def batch_analysis():
    print("=== 开始批量分析 (中文绘图版) ===")
    data = []
    
    if not os.path.exists(MODEL_DIR):
        print(f"错误：未找到文件夹 '{MODEL_DIR}'。")
        return

    files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pt')]
    print(f"找到 {len(files)} 个模型文件。")
    
    # --- 缓存处理 ---
    csv_cache = os.path.join(OUTPUT_DIR, "experiment_summary.csv")
    need_rescan = True

    if os.path.exists(csv_cache):
        try:
            df = pd.read_csv(csv_cache)
            if "Model_Path" in df.columns and "L2_Error" in df.columns:
                print("检测到有效缓存，正在加载...")
                need_rescan = False
            else:
                print("缓存文件过期或不完整，准备重新扫描...")
        except Exception:
            print("缓存文件损坏，准备重新扫描...")
    
    if need_rescan:
        print("正在扫描模型并评估误差 (这可能需要几分钟)...")
        for i, f in enumerate(files):
            params = parse_filename(f)
            if params is None: continue
            
            opt, N, M, lam = params
            path = os.path.join(MODEL_DIR, f)
            
            res = load_and_evaluate(path, resolution=100) 
            
            if res is not None:
                data.append({
                    "Optimizer": opt,
                    "N": N,
                    "M": M,
                    "Lambda": lam,
                    "L2_Error": res["l2_error"],
                    "Model_Path": path
                })
            
            if (i+1) % 20 == 0: print(f"已处理 {i+1} 个文件...")
        
        df = pd.DataFrame(data)
        df.to_csv(csv_cache, index=False)
        print(f"数据已保存至 {csv_cache}")

    if df.empty:
        print("未找到有效数据，请检查模型文件。")
        return

    # --- 绘图 ---
    optimizers = df["Optimizer"].unique()
    print(f"\n检测到的优化器: {optimizers}")

    for opt in optimizers:
        print(f"\n--- 正在处理优化器: {opt} ---")
        
        plot_3d_bar_chart(df, opt)
        
        opt_df = df[df["Optimizer"] == opt]
        if opt_df.empty: continue
            
        best_idx = opt_df["L2_Error"].idxmin()
        best_row = opt_df.loc[best_idx]
        
        print(f"  最佳配置: N={best_row['N']}, M={best_row['M']}, Lambda={best_row['Lambda']}")
        print(f"  最小 L2 误差: {best_row['L2_Error']:.2e}")
        
        full_res = load_and_evaluate(best_row['Model_Path'], resolution=256)
        
        if full_res:
            title_str = f"{opt}, N={best_row['N']}, M={best_row['M']}, $\lambda$={best_row['Lambda']}"
            plot_solution_comparison(
                full_res, 
                title_str, 
                save_name=f"best_solution_{opt}.png"
            )

    print("\n=== 分析完成！ ===")
    print(f"请查看 '{OUTPUT_DIR}' 文件夹获取图像。")

if __name__ == "__main__":
    batch_analysis()