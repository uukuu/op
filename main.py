import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ======================
# 0. 全局设置
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(1234)
np.random.seed(1234)

# ======================
# 1. PDE 精确解 / 源项 / 边界
# ======================

def exact_u(x, y):
    # 用 numpy 计算精确解，只在评估阶段用
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def f_source(x, y):
    # 用 torch 计算右端 f(x,y)，用于训练
    return -2.0 * torch.pi**2 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

def g_boundary(x, y):
    # Dirichlet: u = 0
    return torch.zeros_like(x)

# ======================
# 2. 采样内部 / 边界点
# ======================
def sample_interior(N):
    x = np.random.rand(N, 1)
    y = np.random.rand(N, 1)
    return x, y

def sample_boundary(M):
    M_each = M // 4

    # x=0
    y0 = np.random.rand(M_each, 1)
    x0 = np.zeros_like(y0)
    # x=1
    y1 = np.random.rand(M_each, 1)
    x1 = np.ones_like(y1)
    # y=0
    x2 = np.random.rand(M_each, 1)
    y2 = np.zeros_like(x2)
    # y=1
    x3 = np.random.rand(M_each, 1)
    y3 = np.ones_like(x3)

    x = np.vstack([x0, x1, x2, x3])
    y = np.vstack([y0, y1, y2, y3])
    return x, y

# ======================
# 3. PINN 网络
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
# 4. PINN 损失函数
# ======================
def pinn_loss(model, x_int, y_int, x_b, y_b, lambda_bc):
    x_int.requires_grad_(True)
    y_int.requires_grad_(True)

    # PDE 残差
    u_int = model(x_int, y_int)
    grads = torch.autograd.grad(
        u_int, (x_int, y_int),
        grad_outputs=torch.ones_like(u_int),
        create_graph=True,
        retain_graph=True
    )
    u_x, u_y = grads
    u_xx = torch.autograd.grad(
        u_x, x_int,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True
    )[0]
    u_yy = torch.autograd.grad(
        u_y, y_int,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True,
        retain_graph=True
    )[0]

    f_int = f_source(x_int, y_int)
    r_pde = u_xx + u_yy - f_int
    loss_pde = torch.mean(r_pde**2)

    # 边界残差
    u_b = model(x_b, y_b)
    g_b = g_boundary(x_b, y_b)
    r_bc = u_b - g_b
    loss_bc = torch.mean(r_bc**2)

    loss = loss_pde + lambda_bc * loss_bc
    return loss, loss_pde.detach(), loss_bc.detach()

# ======================
# 5. 通用训练函数：支持 GD / Adam / LBFGS
# ======================
def train_pinn(
    N_int=2000,
    M_b=400,
    hidden_dim=20,
    n_hidden=3,
    lambda_bc=1.0,
    optimizer_type="GD",   # "GD" / "Adam" / "LBFGS"
    lr=1e-3,
    n_epochs=3000,
    print_every=500,
    model_init=None        # 供混合策略使用，这里我们比较三种单独算法用不到
):
    # 采样点
    print(0)
    x_int_np, y_int_np = sample_interior(N_int)
    x_b_np, y_b_np = sample_boundary(M_b)

    x_int = torch.tensor(x_int_np, dtype=torch.float32, device=device)
    y_int = torch.tensor(y_int_np, dtype=torch.float32, device=device)
    x_b   = torch.tensor(x_b_np, dtype=torch.float32, device=device)
    y_b   = torch.tensor(y_b_np, dtype=torch.float32, device=device)

    # 建模 / 或使用给定初值
    if model_init is None:
        model = FCNet(
            in_dim=2, out_dim=1,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden
        ).to(device)
    else:
        model = model_init
    print(1)
    # 选择优化器
    if optimizer_type == "GD":            # 纯梯度下降 = 无动量 SGD
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == "LBFGS":
        optimizer = optim.LBFGS(model.parameters(), lr=lr,
                                max_iter=500, history_size=50)
        print_every=10  # LBFGS epoch 少，调整打印间隔  
    else:
        raise ValueError("Unknown optimizer type")

    history = {"loss": [], "loss_pde": [], "loss_bc": []}
    print(2)
    def closure():
        optimizer.zero_grad()
        loss, loss_pde, loss_bc = pinn_loss(model, x_int, y_int, x_b, y_b, lambda_bc)
        loss.backward()
        return loss

    # 训练循环
    for epoch in range(1, n_epochs + 1):
        # print(epoch,"/",n_epochs)
        if optimizer_type == "LBFGS":
            loss = optimizer.step(closure)
            loss, loss_pde, loss_bc = pinn_loss(model, x_int, y_int, x_b, y_b, lambda_bc)
        else:  # GD / Adam
            optimizer.zero_grad()
            loss, loss_pde, loss_bc = pinn_loss(model, x_int, y_int, x_b, y_b, lambda_bc)
            loss.backward()
            optimizer.step()

        history["loss"].append(loss.item())
        history["loss_pde"].append(loss_pde.item())
        history["loss_bc"].append(loss_bc.item())

        if epoch % print_every == 0:
            print(f"[{optimizer_type}] Epoch {epoch}/{n_epochs}, "
                  f"loss={loss.item():.3e}, pde={loss_pde.item():.3e}, bc={loss_bc.item():.3e}")

    return model, history

# ======================
# 6. 评估：L2 误差 + 可视化
# ======================
def evaluate_model(model, n_test=101, plot=True, title="PINN"):
    x = np.linspace(0, 1, n_test)
    y = np.linspace(0, 1, n_test)
    X, Y = np.meshgrid(x, y)
    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)

    x_t = torch.tensor(x_flat, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_flat, dtype=torch.float32, device=device)

    with torch.no_grad():
        u_pred = model(x_t, y_t).cpu().numpy().reshape(n_test, n_test)
    u_ex = exact_u(X, Y)
    diff = u_pred - u_ex
    l2 = np.sqrt(np.mean(diff**2))
    print(f"{title}: L2 error ≈ {l2:.3e}")

    if plot:
        fig = plt.figure(figsize=(12, 4))
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax1.plot_surface(X, Y, u_ex, linewidth=0, antialiased=False)
        ax1.set_title("Exact")

        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax2.plot_surface(X, Y, u_pred, linewidth=0, antialiased=False)
        ax2.set_title("Predicted")

        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        ax3.plot_surface(X, Y, np.abs(diff), linewidth=0, antialiased=False)
        ax3.set_title("|Error|")

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    return l2

# ======================
# 7. 主程序：GD / Adam / LBFGS 三种方法对比
# ======================
if __name__ == "__main__":
    # 公共超参数
    N_int = 2000
    M_b = 400
    hidden_dim = 20
    n_hidden = 3
    lambda_bc = 1.0

    optimizers = [
        ("GD",    {"lr": 1e-2, "n_epochs": 5000}),  # 步长大一点
        # ("Adam",  {"lr": 1e-3, "n_epochs": 5000}),
        # ("LBFGS", {"lr": 1e-2,  "n_epochs": 50}),    # 外层 epoch 少一些即可
    ]
    import os
    os.makedirs("models", exist_ok=True)   # 存模型
    os.makedirs("results", exist_ok=True)  # 存表格等
    histories = {}
    errors = {}
    all_results = []

    # 为每个优化器分别找最优
    best_per_opt = {
        "GD":    {"L2": float("inf"), "cfg": None},
        "Adam":  {"L2": float("inf"), "cfg": None},
        "LBFGS": {"L2": float("inf"), "cfg": None},
    }
    for opt_name, cfg in optimizers:
        print(f"\n============================")
        print(f"   Optimizer = {opt_name}")
        print(f"============================")
        for N_int in [400,1000,2000,5000]:
            for M_b in [100,200,400,1000]:
                for lambda_bc in [0.1,1.0,10.0,50.0]:
                    print(f"\n--- Training {opt_name}: N={N_int}, M={M_b}, λ={lambda_bc} ---")
    # 逐个训练 + 评估
                    torch.manual_seed(1025)       # 保证初始化完全一样
                    np.random.seed(1025)
                    model, hist = train_pinn(
                        N_int=N_int,
                        M_b=M_b,
                        hidden_dim=hidden_dim,
                        n_hidden=n_hidden,
                        lambda_bc=lambda_bc,
                        optimizer_type=opt_name,
                        lr=cfg["lr"],
                        n_epochs=cfg["n_epochs"],
                        print_every=max(1, cfg["n_epochs"] // 10)
                    )
                    l2_err = evaluate_model(
                        model,
                        plot=False,            # 调参阶段不画图
                        title=f"{opt_name}-N{N_int}-M{M_b}-lam{lambda_bc}"
                    )

                    # =====================
                    # 1) 保存模型
                    # =====================
                    lambda_str = str(lambda_bc).replace(".", "p")
                    model_name = f"{opt_name}-{N_int}-{M_b}-{lambda_str}.pt"
                    model_path = os.path.join("models", model_name)
                    torch.save(model.state_dict(), model_path)
                    print(f"Model saved to {model_path}")

                    # =====================
                    # 2) 存储结果
                    # =====================
                    all_results.append({
                        "opt": opt_name,
                        "N": N_int,
                        "M": M_b,
                        "lambda": lambda_bc,
                        "L2": l2_err,
                        "model": model_path
                    })

                    # =====================
                    # 3) 对当前优化器更新最优
                    # =====================
                    if l2_err < best_per_opt[opt_name]["L2"]:
                        best_per_opt[opt_name]["L2"] = l2_err
                        best_per_opt[opt_name]["cfg"] = {
                            "opt": opt_name,
                            "N": N_int,
                            "M": M_b,
                            "lambda": lambda_bc,
                            "model": model_path
                        }

    # ---------- 可视化 1：损失收敛曲线 ----------
    plt.figure(figsize=(6, 4))
    for opt_name, hist in histories.items():
        plt.semilogy(hist["loss"], label=opt_name)
    plt.xlabel("epoch")
    plt.ylabel("loss (log scale)")
    plt.title("Training loss comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------- 可视化 2：L2 误差柱状图 ----------
    plt.figure(figsize=(6, 4))
    names = list(errors.keys())
    vals = [errors[k] for k in names]
    plt.bar(names, vals)
    plt.ylabel("L2 error")
    plt.title("L2 error of different optimizers")
    for i, v in enumerate(vals):
        plt.text(i, v, f"{v:.2e}", ha="center", va="bottom", fontsize=8, rotation=0)
    plt.tight_layout()
    plt.show()

    # ---------- 可视化 3：选一个最优算法画解的三维图 ----------
    # 这里选 L2 最小的算法
    exit()
    best_name = min(errors, key=errors.get)
    print("\nBest optimizer (by L2 error):", best_name)
    # 再训练一次（或你也可以在上面保存对应 model，这里简单再跑一遍）
    model_best, _ = train_pinn(
        N_int=N_int,
        M_b=M_b,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        lambda_bc=lambda_bc,
        optimizer_type=best_name,
        lr=optimizers[[o[0] for o in optimizers].index(best_name)][1]["lr"],
        n_epochs=optimizers[[o[0] for o in optimizers].index(best_name)][1]["n_epochs"],
        print_every=999999  # 不打印
    )
    evaluate_model(model_best, plot=True, title=f"Best optimizer: {best_name}")
