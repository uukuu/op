import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import matplotlib.font_manager as fm
import platform

# ======================
# 0. 全局设置与绘图修复
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 设置基础绘图风格
plt.style.use('seaborn-v0_8-whitegrid')

# 2. 强制解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm' # 数学公式字体

# 3. 智能字体加载系统
def set_chinese_font():
    """自动寻找并加载系统中的中文字体"""
    system_name = platform.system()
    font_files = []
    if system_name == 'Windows':
        font_files = ['simhei.ttf', 'msyh.ttc', 'simsun.ttc', 'kaiu.ttf']
    elif system_name == 'Darwin': # Mac
        font_files = ['Arial Unicode.ttf', 'PingFang.ttc', 'Heiti.ttc']
    else: # Linux
        font_files = ['wqy-microhei.ttc', 'wqy-zenhei.ttc', 'DroidSansFallback.ttf']
        
    for f in fm.fontManager.ttflist:
        if any(x in f.name for x in ['SimHei', 'Microsoft YaHei', 'Heiti', 'WenQuanYi']):
            plt.rcParams['font.sans-serif'] = [f.name, 'Arial']
            print(f"✅ 成功匹配到中文字体: {f.name}")
            return True

    print("⚠️ 未在缓存中找到常用中文字体，正在搜索系统目录...")
    font_dirs = ['C:\\Windows\\Fonts', '/Library/Fonts', '/System/Library/Fonts', '/usr/share/fonts']
    for font_dir in font_dirs:
        if not os.path.exists(font_dir): continue
        for filename in font_files:
            file_path = os.path.join(font_dir, filename)
            if os.path.exists(file_path):
                prop = fm.FontProperties(fname=file_path)
                print(f"✅ 已强制加载字体文件: {file_path}")
                plt.rcParams['font.family'] = prop.get_name()
                return True
    return False

HAS_CHINESE = set_chinese_font()

def get_text(cn, en):
    return cn if HAS_CHINESE else en

torch.manual_seed(1234)
np.random.seed(1234)

# ======================
# 1. PDE 定义
# ======================
def exact_u(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def f_source(x, y):
    return -2.0 * torch.pi**2 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

def g_boundary(x, y):
    return torch.zeros_like(x)

# ======================
# 2. 采样函数
# ======================
def sample_interior(N):
    x = np.random.rand(N, 1)
    y = np.random.rand(N, 1)
    return x, y

def sample_boundary(M):
    M_each = M // 4
    x1 = np.random.rand(M_each, 1); y1 = np.zeros_like(x1)
    x2 = np.random.rand(M_each, 1); y2 = np.ones_like(x2)
    y3 = np.random.rand(M_each, 1); x3 = np.zeros_like(y3)
    y4 = np.random.rand(M_each, 1); x4 = np.ones_like(y4)
    x = np.vstack([x1, x2, x3, x4])
    y = np.vstack([y1, y2, y3, y4])
    return x, y

# ======================
# 3. 网络架构
# ======================
class FCNet(nn.Module):
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

# ======================
# 4. Loss 计算
# ======================
def pinn_loss(model, x_int, y_int, x_b, y_b, lambda_bc):
    # 必须开启梯度追踪，否则无法求导
    if not x_int.requires_grad: x_int.requires_grad_(True)
    if not y_int.requires_grad: y_int.requires_grad_(True)

    # PDE Residual
    u_int = model(x_int, y_int)
    grads = torch.autograd.grad(u_int, (x_int, y_int), grad_outputs=torch.ones_like(u_int), create_graph=True, retain_graph=True)
    u_x, u_y = grads[0], grads[1]
    u_xx = torch.autograd.grad(u_x, x_int, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_int, grad_outputs=torch.ones_like(u_y), create_graph=True, retain_graph=True)[0]
    
    f_val = f_source(x_int, y_int)
    loss_pde = torch.mean((u_xx + u_yy - f_val)**2)

    # BC Residual
    u_b = model(x_b, y_b)
    g_val = g_boundary(x_b, y_b)
    loss_bc = torch.mean((u_b - g_val)**2)

    loss = loss_pde + lambda_bc * loss_bc
    return loss, loss_pde, loss_bc

# ======================
# 5. 训练与绘图流程
# ======================
def train_and_plot():
    print("=== 单次 PINN 实验与 Loss 绘图 ===")
    
    opt_input = input("优化器 (GD/Adam/LBFGS) [默认 LBFGS]: ").strip()
    optimizer_name = opt_input if opt_input else "LBFGS"
    
    N_input = input("内部采样点数 N [默认 2000]: ").strip()
    N = int(N_input) if N_input else 2000
    
    M_input = input("边界采样点数 M [默认 400]: ").strip()
    M = int(M_input) if M_input else 400
    
    lam_input = input("惩罚系数 Lambda [默认 1.0]: ").strip()
    lambda_bc = float(lam_input) if lam_input else 1.0

    # 参数设置
    if optimizer_name == "LBFGS":
        lr = 1.0; epochs = 100; print_interval = 10
    elif optimizer_name == "Adam":
        lr = 1e-3; epochs = 5000; print_interval = 500
    else: 
        lr = 1e-2; epochs = 10000; print_interval = 1000

    print(f"\n>>> 启动训练: {optimizer_name}, N={N}, M={M}, Lambda={lambda_bc}")
    
    x_int, y_int = sample_interior(N)
    x_b, y_b = sample_boundary(M)
    
    x_int = torch.tensor(x_int, dtype=torch.float32, device=device)
    y_int = torch.tensor(y_int, dtype=torch.float32, device=device)
    x_b = torch.tensor(x_b, dtype=torch.float32, device=device)
    y_b = torch.tensor(y_b, dtype=torch.float32, device=device)

    model = FCNet().to(device)

    if optimizer_name == "LBFGS":
        optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=500, line_search_fn="strong_wolfe")
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)

    history = {'loss': [], 'pde': [], 'bc': []}
    start_time = time.time()

    def closure():
        optimizer.zero_grad()
        loss, _, _ = pinn_loss(model, x_int, y_int, x_b, y_b, lambda_bc)
        loss.backward()
        return loss

    for epoch in range(1, epochs + 1):
        if optimizer_name == "LBFGS":
            optimizer.step(closure)
        else:
            optimizer.zero_grad()
            loss, _, _ = pinn_loss(model, x_int, y_int, x_b, y_b, lambda_bc)
            loss.backward()
            optimizer.step()
        
        # [修复] 移除了 with torch.no_grad(): 
        # 但使用 .item() 来取值，既不会报错，也不会导致显存泄露
        l_total, l_pde, l_bc = pinn_loss(model, x_int, y_int, x_b, y_b, lambda_bc)
        history['loss'].append(l_total.item())
        history['pde'].append(l_pde.item())
        history['bc'].append(l_bc.item())

        if epoch % print_interval == 0:
            print(f"Epoch {epoch}/{epochs} | Loss: {l_total.item():.3e}")

    print(f"训练结束，耗时: {time.time() - start_time:.2f}s")

    os.makedirs("models", exist_ok=True)
    save_name = f"models/manual_{optimizer_name}_N{N}_M{M}_lam{lambda_bc}.pt"
    torch.save(model.state_dict(), save_name)
    
    # 绘图
    plot_loss_curve(history, optimizer_name, N, M, lambda_bc)
    evaluate_model(model)

def plot_loss_curve(history, opt_name, N, M, lam):
    """绘制 Loss 收敛曲线"""
    fig, ax = plt.figure(figsize=(10, 6), dpi=150), plt.gca()
    
    iterations = np.arange(1, len(history['loss']) + 1)
    
    l_total = get_text('Total Loss (总误差)', 'Total Loss')
    l_pde = get_text('PDE Loss (物理残差)', 'PDE Loss')
    l_bc = get_text('BC Loss (边界残差)', 'BC Loss')
    x_label = get_text('迭代次数 (Iterations)', 'Iterations / Epochs')
    title = get_text(f'Loss 收敛曲线 ({opt_name}, N={N}, M={M}, $\lambda$={lam})', 
                     f'Loss Convergence ({opt_name}, N={N}, M={M}, $\lambda$={lam})')

    ax.semilogy(iterations, history['loss'], 'k-', linewidth=2, label=l_total)
    ax.semilogy(iterations, history['pde'], 'b--', linewidth=1.5, alpha=0.7, label=l_pde)
    ax.semilogy(iterations, history['bc'], 'r:', linewidth=1.5, alpha=0.8, label=l_bc)
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Loss (Log Scale)', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    ax.legend(fontsize=10, frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    
    os.makedirs("results", exist_ok=True)
    save_path = f"results/loss_curve_{opt_name}_manual.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Loss 曲线图已保存至: {save_path}")
    plt.show()

def evaluate_model(model):
    """计算相对 L2 误差"""
    model.eval()
    n_test = 100
    x = np.linspace(0, 1, n_test)
    y = np.linspace(0, 1, n_test)
    X, Y = np.meshgrid(x, y)
    x_t = torch.tensor(X.reshape(-1, 1), dtype=torch.float32).to(device)
    y_t = torch.tensor(Y.reshape(-1, 1), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        u_pred = model(x_t, y_t).cpu().numpy().reshape(X.shape)
    u_real = exact_u(X, Y)
    error = np.linalg.norm(u_pred - u_real) / np.linalg.norm(u_real)
    print(f"📊 最终相对 L2 误差: {error:.2e}")

if __name__ == "__main__":
    train_and_plot()