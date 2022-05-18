import torch
import matplotlib.pyplot as plt
import numpy as np
import progressbar
import copy

def angle(vec1, vec2):
    return torch.acos(torch.dot(vec1, vec2)/(vec1.norm()*vec2.norm())).item()


def rad2deg(angle):
    return angle*180/np.pi


def concat_torch_list(torch_list):
    for i, t in enumerate(torch_list):
        torch_list[i] = t.flatten()
    return torch.cat(torch_list)


def create_random_directions(weights, ignore1D=False, seed=42, device="cpu"):
    torch.manual_seed(seed)
    direction = [torch.randn(w.size()).to(device) for w in weights]
    
    # apply filter normalisation, where every perturbation d in direction has the same norm as its corresponding w in weights
    for d, w in zip(direction, weights):
        if ignore1D and d.dim() <= 1:
            d.fill_(0)
        d.mul_(w.norm()/(d.norm() + 1e-10)) # add small perturbation to avoid division by zero

    return direction


def update_weights(model, origin_weights, x_dir, y_dir, dx=0.1, dy=0.1, device="cpu"):
    updates = [x.to(device)*dx + y.to(device)*dy for (x, y) in zip(x_dir, y_dir)]
    for (p, w, u) in zip(model.parameters(), origin_weights, updates):
        p.data = w + u
    return None


def plot_loss_landscape(xx, yy, loss_landscape, vmin=0, vmax=20):
    fig, ax = plt.subplots(figsize=(8, 8),subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xx.numpy(), yy.numpy(), loss_landscape.numpy(), cmap='viridis', edgecolor='none', linewidth=0, antialiased=True,  rstride=1, cstride=1, vmin=vmin, vmax=vmax)
    ax.set_xlabel(r'X')
    ax.set_ylabel(r'Y')
    ax.set_zlabel(r'Loss')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def contour_loss_landscape(xx, yy, loss_landscape, vmin=0, vmax=20):
    fig, ax = plt.subplots(figsize=(7, 7))
    surf = ax.contour(xx, yy, loss_landscape, cmap='viridis', levels=20, vmin=vmin, vmax=vmax)
    ax.set_xlabel(r'X')
    ax.set_ylabel(r'Y')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def eval(model, criterion, dataloader, device="cpu"):
    model.eval()
    total_loss = 0.
    for i, (X, _) in enumerate(dataloader):
        with torch.no_grad():
            X = X.to(device)
            output = model(X)
            loss = criterion(output, X)
            total_loss += loss*X.size(0)
    return total_loss/len(dataloader.dataset)


def compute_loss_landscape(model, data_loader, criterion, xmin=-0.5, xmax=0.5, ymin=-0.5, ymax=0.5, device="cpu"):
    # Get weights at origin (trained model weights)
    model = model.to(device)
    weights = [p.data for p in model.parameters()]

    # Get 2 random perturbation directions
    x_dir = create_random_directions(weights, ignore1D=True, seed=42, device=device)
    y_dir = create_random_directions(weights, ignore1D=True, seed=23, device=device)

    # Check directions are orthogonal
    x_vec, y_vec = concat_torch_list(copy.copy(x_dir)), concat_torch_list(copy.copy(y_dir))
    print("Angle between x_dir and y_dir: %.2f °"%rad2deg(angle(x_vec, y_vec)))

    # Create arrays for storing perturbation distances and losses
    nx, ny = 25, 25
    dx_arr = torch.linspace(xmin, xmax, nx)
    dy_arr = torch.linspace(ymin, ymax, ny)
    xx, yy = torch.meshgrid(dx_arr, dy_arr)
    loss_landscape = torch.zeros_like(xx) - 1

    # Evaluate loss perturbing the weights
    model_to_perturb = copy.deepcopy(model)
    with progressbar.ProgressBar(max_value=nx*ny) as bar:
      for i, dx in enumerate(dx_arr):
          for j, dy in enumerate(dy_arr):
              # Perturb weights and evaluate loss
              update_weights(model_to_perturb, weights, x_dir, y_dir, dx, dy, device)
              loss = eval(model_to_perturb, criterion, data_loader, device)
              loss_landscape[i, j] = loss.item()

              # Ocasionally print loss
            #   if (nx*i + j) % 10 == 0:
            #     print("\t Loss: ", loss)

              bar.update(nx*i + j)
    return xx, yy, loss_landscape

def visualise_landscape(model, loader, criterion, xmin=-0.5, xmax=0.5, ymin=-0.5, ymax=0.5, vmin=None, vmax=None, mode="plot", device="cpu"):
    xx, yy, loss_landscape = compute_loss_landscape(model, loader, criterion, xmin, xmax, ymin, ymax, device=device)
    if mode=="contour":
        contour_loss_landscape(xx, yy, loss_landscape, vmin, vmax)
    elif mode=="plot" or mode=="3d":
        plot_loss_landscape(xx, yy, loss_landscape, vmin, vmax)
    else:
        pass
    return xx, yy, loss_landscape


def path_in_landscape(loss_landscape, xx, yy, model):
    model_to_perturb = copy.deepcopy(model)
    # for epoch
        # train model
        # get loss
        # find x, y in landscape

    weights = [p.data for p in model_to_perturb.parameters()]