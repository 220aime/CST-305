
# Project 5 – Self-Organized Criticality
# Aime Serge Tuyishime
# Professor Ricardo Citro
# March 22. 2025
# Principles of Modeling and Simulation Lecture & Lab | CST-305
# This simulation models dynamic memory fragmentation
# over time using file save/delete operations. It also
# integrates deterministic chaos using the Lorenz system.

import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation


# Class to simulate file storage and fragmentation behavior
class StorageSimulator:
    def __init__(self, size):
        self.size = size  # Total memory size
        self.free_blocks = [(0, size - 1)]  # Initial block is all free
        self.files = {}  # Dictionary to store file fragments
        self.next_file_id = 0  # Incremental ID for new files
        self.frag_history = []  # Fragmentation level over time
        self.save_times = []  # Time taken to save files
        self.access_times = []  # Time taken to access files
        self.assembly_times = []  # Time taken to assemble deleted files
        self.threshold = 0.7  # Critical fragmentation threshold

    # Save file by splitting it across available blocks
    def save_file(self, size, chaos_factor):
        total_free = sum(end - start + 1 for start, end in self.free_blocks)
        if total_free < size:
            return False  # Not enough space

        # Sort free blocks by size (largest first)
        free_sorted = sorted(self.free_blocks, key=lambda x: -(x[1] - x[0] + 1))
        remaining = size
        fragments = []

        for block in free_sorted:
            if remaining <= 0:
                break
            block_size = block[1] - block[0] + 1
            alloc_size = min(remaining, block_size)
            fragments.append((block[0], block[0] + alloc_size - 1))
            remaining -= alloc_size
            # Update free blocks
            if alloc_size < block_size:
                self.free_blocks.append((block[0] + alloc_size, block[1]))
            self.free_blocks.remove(block)

        if remaining == 0:
            self.files[self.next_file_id] = fragments  # Save file fragments
            self.next_file_id += 1
            self.free_blocks.sort()
            self.merge_free()

            # Simulate save time based on fragmentation and chaos factor
            frag = self.calculate_fragmentation()
            save_time = 1 + chaos_factor * frag
            self.save_times.append(save_time)

            return True
        return False

    # Delete a file and reclaim its space
    def delete_file(self, file_id, chaos_factor_y, chaos_factor_z):
        if file_id not in self.files:
            return False
        for fragment in self.files[file_id]:
            self.free_blocks.append(fragment)
        del self.files[file_id]
        self.merge_free()

        # Simulate access and assembly time based on fragmentation
        frag = self.calculate_fragmentation()
        access_time = 1 + chaos_factor_y * frag
        assembly_time = 1 + chaos_factor_z * frag
        self.access_times.append(access_time)
        self.assembly_times.append(assembly_time)

        return True

    # Merge contiguous free blocks
    def merge_free(self):
        self.free_blocks.sort()
        merged = []
        for block in self.free_blocks:
            if not merged:
                merged.append(block)
            else:
                last = merged[-1]
                if block[0] == last[1] + 1:
                    merged[-1] = (last[0], block[1])  # Merge adjacent blocks
                else:
                    merged.append(block)
        self.free_blocks = merged

    # Calculate average fragmentation per file
    def calculate_fragmentation(self):
        total_frags = sum(len(frags) for frags in self.files.values())
        num_files = len(self.files)
        return total_frags / num_files if num_files > 0 else 0

    # Perform one simulation step (save or delete)
    def step(self, action, x, y, z):
        frag = self.calculate_fragmentation()
        if action == 'save':
            size = random.randint(10, 50)
            return self.save_file(size, abs(x) % 1)
        elif action == 'delete' and self.files:
            file_id = random.choice(list(self.files.keys()))
            return self.delete_file(file_id, abs(y) % 1, abs(z) % 1)
        return False


# Lorenz system equations to model chaos
def lorenz(x, y, z, s=10, r=28, b=2.667):
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return x_dot, y_dot, z_dot


# --- Simulation Setup ---

storage_size = 200  # Memory size
sim_steps = 500  # Total simulation steps
save_prob = 0.6  # Probability to perform save over delete
sim = StorageSimulator(storage_size)

# Initial Lorenz values and step size
dt = 0.02
xs, ys, zs = [0.1], [0.0], [0.0]

# --- Visualization Setup ---

fig = plt.figure(figsize=(14, 12))

# Subplot for memory view
ax1 = plt.subplot2grid((4, 2), (0, 0), colspan=2)
# Subplot for fragmentation chart
ax_frag = plt.subplot2grid((4, 2), (1, 0), colspan=2)
# Subplots for time metrics
ax2 = plt.subplot2grid((4, 2), (2, 0))  # Save time
ax3 = plt.subplot2grid((4, 2), (2, 1))  # Access & assembly
# Subplot for Lorenz 3D trajectory
ax4 = plt.subplot2grid((4, 2), (3, 0), colspan=2, projection='3d')

# Display memory blocks as a bar
memory_bar = ax1.imshow(np.zeros((1, storage_size)), aspect='auto', cmap='tab10')
ax1.set_title('Memory Fragmentation')

# Fragmentation line chart
frag_line, = ax_frag.plot([], [], color='red')
ax_frag.set_title('Fragmentation Over Time')
ax_frag.set_ylim(0, 2)

# Save time line
save_line, = ax2.plot([], [], color='green')
ax2.set_title('Save Time')

# Access and assembly times
access_line, = ax3.plot([], [], color='purple')
assembly_line, = ax3.plot([], [], color='orange')
ax3.set_title('Access & Assembly Time')

# Lorenz trajectory
lorenz_line, = ax4.plot(xs, ys, zs, lw=0.5)


# --- Animation Update Function ---

def update(frame):
    global xs, ys, zs

    x, y, z = xs[-1], ys[-1], zs[-1]

    # Choose action based on probability and current state
    action = 'save' if random.random() < save_prob else 'delete' if sim.files else 'save'
    sim.step(action, x, y, z)

    # Update fragmentation
    frag = sim.calculate_fragmentation()
    sim.frag_history.append(frag)

    # Update Lorenz values
    dx, dy, dz = lorenz(x, y, z)
    xs.append(x + dx * dt)
    ys.append(y + dy * dt)
    zs.append(z + dz * dt)

    # Update memory view
    mem = np.zeros(storage_size)
    for file_id, fragments in sim.files.items():
        for start, end in fragments:
            mem[start:end + 1] = file_id % 10 + 1
    memory_bar.set_data([mem])

    # Update fragmentation chart
    frag_line.set_data(range(len(sim.frag_history)), sim.frag_history)
    ax_frag.set_xlim(0, max(100, len(sim.frag_history)))

    # Update time plots
    save_line.set_data(range(len(sim.save_times)), sim.save_times)
    access_line.set_data(range(len(sim.access_times)), sim.access_times)
    assembly_line.set_data(range(len(sim.assembly_times)), sim.assembly_times)

    ax2.set_xlim(0, max(100, len(sim.save_times)))
    ax2.set_ylim(0, max(3, max(sim.save_times, default=1)))
    ax3.set_xlim(0, max(100, len(sim.access_times)))
    ax3.set_ylim(0, max(3, max(sim.access_times + sim.assembly_times, default=1)))

    # Update Lorenz 3D plot
    lorenz_line.set_data(xs, ys)
    lorenz_line.set_3d_properties(zs)
    ax4.auto_scale_xyz(xs, ys, zs)

    # Display alert if fragmentation is critical
    if frag >= sim.threshold:
        ax1.set_title('CRITICAL THRESHOLD REACHED!')

    return memory_bar, frag_line, save_line, access_line, assembly_line, lorenz_line


# Run animation
ani = FuncAnimation(fig, update, frames=sim_steps, blit=False, interval=100)
plt.tight_layout()
plt.show()
