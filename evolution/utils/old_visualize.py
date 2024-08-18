from collections import defaultdict
from typing import Union
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
from IPython.display import clear_output
import torch
torch.set_grad_enabled(False)

 
from evolution.core.gworld import GWorld


def visualize(world: GWorld, collisions: Union[None, torch.Tensor], show_rays=True, legend=False):
    clear_output(wait=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Move the food grid to CPU if it's on GPU
    food_grid_cpu = world.food_grid.cpu().numpy()
    
    # Plot the food grid heatmap
    heatmap = ax.imshow(food_grid_cpu, interpolation='nearest', alpha=0.2, cmap='Greens')
    fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    
    # Move creature attributes to CPU if they're on GPU
    positions_cpu = world.creatures.positions.cpu().numpy() + world.creatures.pad  # add pad so it lines up with food_grid
    sizes_cpu = world.creatures.sizes.cpu().numpy()
    head_dirs_cpu = world.creatures.head_dirs.cpu().numpy()
    colors_cpu = world.creatures.colors.cpu().numpy()
    rays_cpu = world.creatures.rays.cpu().numpy()

    for i in range(len(positions_cpu)):
        pos = positions_cpu[i]
        size = sizes_cpu[i]
        color = colors_cpu[i] / 255  # Convert to 0-1 range for matplotlib
        head_dir = head_dirs_cpu[i]
        head_pos = pos + head_dir * size
        
        # Plot the creature as a circle
        circle = plt.Circle(pos, size, color=color, fill=True, alpha=0.9, label=i)
        ax.add_patch(circle)
        
        # Plot the head direction
        ax.plot([pos[0], head_pos[0]], [pos[1], head_pos[1]], color='black')
        
        if show_rays:
            # Plot the rays
            if collisions is not None:
                collisions_cpu = collisions.cpu().numpy()
                for j in range(len(rays_cpu[i])):
                    ray_dir = rays_cpu[i][j][:2]
                    ray_len = rays_cpu[i][j][2]
                    ray_end = pos + ray_dir * ray_len
                    collision_info = collisions_cpu[i][j]
                    if np.any(collision_info[:3] != 0):  # If any component is not zero, ray is active
                        ray_color = collision_info / 255  # Convert to 0-1 range for matplotlib
                        ax.plot([pos[0], ray_end[0]], [pos[1], ray_end[1]], color=ray_color)
                    else:
                        ax.plot([pos[0], ray_end[0]], [pos[1], ray_end[1]], color='gray', alpha=0.3)
            else:
                for j in range(len(rays_cpu[i])):
                    ray_dir = rays_cpu[i][j][:2]
                    ray_len = rays_cpu[i][j][2]
                    ray_end = pos + ray_dir * ray_len
                    ax.plot([pos[0], ray_end[0]], [pos[1], ray_end[1]], color='gray', alpha=0.3)
    
    ax.set_xlim([0, food_grid_cpu.shape[1]])
    ax.set_ylim([0, food_grid_cpu.shape[0]])
    ax.set_aspect('equal')
    plt.xticks([])
    plt.yticks([])
    if legend:
        plt.legend()
    plt.gca().invert_yaxis()
    plt.title(f'Step {world.time} - Population {world.population}')
    plt.show()


def visualize_grid_setup(world: GWorld, cells: torch.Tensor, cell_counts: torch.Tensor, 
                         collisions: Union[None, torch.Tensor]=None, show_rays=True):
    """Visualizes the grid setup for ray tracing."""
    # match the return order from self.compute_grid_setup for ease of use
    num_objects = world.population
    num_cells = cells.shape[0]
    width = num_cells * world.cfg.cell_size

    positions = world.creatures.positions.cpu().numpy()
    sizes = world.creatures.sizes.cpu().numpy()
    # print(sizes)
    colors = world.creatures.colors.cpu().numpy()
    
    for obj_idx in range(num_objects):
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create a heatmap for the current object
        heatmap = np.ones((num_cells, num_cells, 3), dtype=int)*255
        
        for y in range(num_cells):
            for x in range(num_cells):
                for k in range(cell_counts[y, x]):
                    if cells[y, x, k] == obj_idx:
                        heatmap[y, x] = colors[obj_idx]
        
        cax = ax.imshow(heatmap[::-1], cmap='viridis', extent=[0, width, 0, width])
        # plt.colorbar(cax, ax=ax)
        
        # Plot the circles
        for i in range(num_objects):
            x, y = positions[i]
            radius = sizes[i]
            color = colors[i] / 255  # Convert to [0, 1] range for Matplotlib
            circle = patches.Circle(
                (x, y),
                radius,
                linewidth=0.4,
                edgecolor='black',
                facecolor=color,
                alpha=0.5
            )
            ax.add_patch(circle)

            if show_rays:
                rays_cpu = world.creatures.rays.cpu().numpy()
                pos = np.array([x, y])
                # Plot the rays
                if collisions is not None:
                    collisions_cpu = collisions.cpu().numpy()
                    for j in range(len(rays_cpu[i])):
                        ray_dir = rays_cpu[i][j][:2]
                        ray_len = rays_cpu[i][j][2]
                        ray_end = pos + ray_dir * ray_len
                        collision_info = collisions_cpu[i][j]
                        if np.any(collision_info[:3] != 0):  # If any component is not zero, ray is active
                            ray_color = collision_info / 255  # Convert to 0-1 range for matplotlib
                            ax.plot([pos[0], ray_end[0]], [pos[1], ray_end[1]], color=ray_color)
                        else:
                            ax.plot([pos[0], ray_end[0]], [pos[1], ray_end[1]], color='gray', alpha=0.3)
                else:
                    for j in range(len(rays_cpu[i])):
                        ray_dir = rays_cpu[i][j][:2]
                        ray_len = rays_cpu[i][j][2]
                        ray_end = pos + ray_dir * ray_len
                        ax.plot([pos[0], ray_end[0]], [pos[1], ray_end[1]], color='gray', alpha=0.3)
        
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
        plt.title(f'Grid Setup Visualization for Object {obj_idx}')
        # show grid lines
        ax.set_xticks(np.arange(0, width, world.cfg.cell_size))#, minor=True)
        ax.set_yticks(np.arange(0, width, world.cfg.cell_size))#, minor=True)
        ax.grid(which='both', color='black', linestyle='-', linewidth=1)
        plt.gca().invert_yaxis()  # Invert y-axis to match array index representation
        plt.show()


def plotly_visualize_grid_setup(world: GWorld, cells: torch.Tensor, cell_counts: torch.Tensor):        
    """Visualizes the grid setup for ray tracing."""
    # match the return order from self.compute_grid_setup for ease of use
    num_objects = world.population
    num_cells = cells.shape[0]
    width = num_cells * world.cfg.cell_size

    positions = world.creatures.positions.cpu().numpy() + world.creatures.pad  # add pad so it lines up with food_grid
    sizes = world.creatures.sizes.cpu().numpy()
    colors = world.creatures.colors.cpu().numpy()
    
    for obj_idx in range(num_objects):
        heatmap = np.ones((num_cells, num_cells, 3), dtype=int) * 255

        for y in range(num_cells):
            for x in range(num_cells):
                for k in range(cell_counts[y, x]):
                    if cells[y, x, k] == obj_idx:
                        heatmap[y, x] = colors[obj_idx]

        fig = go.Figure()

        # Create the heatmap for the current object
        fig.add_trace(go.Image(
            z=heatmap,
            opacity=0.5,
            x0=world.cfg.cell_size/2,
            dx=world.cfg.cell_size,
            y0=world.cfg.cell_size/2,
            dy=world.cfg.cell_size,
        ))

        # Plot the circles
        for i in range(num_objects):
            x, y = positions[i]
            radius = sizes[i]
            color = 'rgba({}, {}, {}, 0.5)'.format(*colors[i])

            min_x = x - radius
            max_x = x + radius
            min_y = y - radius
            max_y = y + radius

            fig.add_shape(
                type='circle',
                xref='x',
                yref='y',
                x0=min_x,
                y0=min_y,
                x1=max_x,
                y1=max_y,
                line=dict(
                    color='black',
                    width=1,
                ),
                fillcolor=color
            )

        fig.update_layout(
            title=f'Grid Setup Visualization for Object {obj_idx}',
            xaxis=dict(
                title='Grid X',
                tickmode='array',
                tickvals=np.arange(0, width, world.cfg.cell_size),
                showgrid=True,
                gridcolor='black',
                gridwidth=None,
                range=[0, width]
            ),
            yaxis=dict(
                title='Grid Y',
                tickmode='array',
                tickvals=np.arange(0, width, world.cfg.cell_size),
                showgrid=True,
                gridcolor='black',
                gridwidth=None,
                scaleanchor=None,
                scaleratio=None,
                range=[0, width]
            ),
            plot_bgcolor='rgba(255,255,255,1)',
            margin=dict(l=0, r=0, t=35, b=0),  # Reduce margins
            height=800,  # Set the height of the figure
        )

        fig.show()