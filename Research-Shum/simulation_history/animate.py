import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

def create_animation_from_files(history_dir="simulation_history"):
    """
    Loads Kirchhoff rod simulation data from text files and creates a 3D animation.

    Args:
        history_dir (str): The directory where the simulation history
                           files are stored.
    """
    print(f"Looking for simulation data in: '{history_dir}'")

    # --- 1. Check if the directory and necessary files exist ---
    # if not os.path.isdir(history_dir):
    #     print(f"Error: History directory '{history_dir}' not found.")
    #     print("Please run the main simulation script first to generate the data.")
    #     return

    params_path = "simulation_history\simulation_params.txt"
    x_history_path = "simulation_history\history_X.txt"

    if not os.path.exists(params_path) or not os.path.exists(x_history_path):
        print("Error: Missing 'simulation_params.txt' or 'history_X.txt'.")
        print("Ensure the simulation completed and saved the files correctly.")
        return

    # --- 2. Load Simulation Parameters ---
    try:
        params = {}
        with open(params_path, 'r') as f:
            for line in f:
                key, value = line.strip().split(' = ')
                # Attempt to convert value to a number (int or float)
                try:
                    if '.' in value or 'e' in value:
                        params[key] = float(value)
                    else:
                        params[key] = int(value)
                except ValueError:
                    params[key] = value # Keep as string if conversion fails

        num_frames = int(params['num_frames'])
        M = int(params['M'])
        dt_snapshot = float(params['dt_snapshot'])
        scenario = params.get('scenario', 'Unknown') # .get() is safer
        L_eff = params.get('L_eff', 1.0)

        print("Successfully loaded parameters:")
        print(f"  Frames: {num_frames}, Points per rod (M): {M}, Scenario: '{scenario}'")

    except Exception as e:
        print(f"Error reading or parsing '{params_path}': {e}")
        return

    # --- 3. Load Position History Data ---
    try:
        print(f"Loading position data from '{x_history_path}'...")
        # Load the data, skipping the header row
        flat_x_data = np.loadtxt(x_history_path, skiprows=1)
        # Reshape the data back to its original 3D structure: (frames, points, coordinates)
        history_X = flat_x_data.reshape(num_frames, M, 3)
        print("Data loaded and reshaped successfully.")
    except Exception as e:
        print(f"Error loading or reshaping data from '{x_history_path}': {e}")
        return

    # --- 4. Set up the 3D Plot ---
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    line, = ax.plot([], [], [], 'o-', lw=2, markersize=4, color='dodgerblue')

    # Determine plot limits to fit the entire animation
    all_coords = history_X.reshape(-1, 3) # Flatten to easily find min/max
    center = np.mean(all_coords, axis=0)
    max_range = np.max(np.ptp(all_coords, axis=0)) # ptp = peak-to-peak (max-min)
    plot_range = max_range * 0.7 if max_range > 0 else L_eff

    ax.set_xlim(center[0] - plot_range, center[0] + plot_range)
    ax.set_ylim(center[1] - plot_range, center[1] + plot_range)
    ax.set_zlim(center[2] - plot_range, center[2] + plot_range)

    ax.set_xlabel("X (um)")
    ax.set_ylabel("Y (um)")
    ax.set_zlabel("Z (um)")
    ax.set_title(f"Kirchhoff Rod Replay: Scenario '{scenario}'")
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, color='black')
    fig.patch.set_facecolor('white') # Ensure figure background is white

    # --- 5. Define the Animation Update Function ---
    def update_animation(frame):
        """
        Updates the plot for each frame of the animation.
        """
        X_data = history_X[frame]
        line.set_data(X_data[:, 0], X_data[:, 1])
        line.set_3d_properties(X_data[:, 2])
        sim_time = frame * dt_snapshot
        time_text.set_text(f"Time: {sim_time:.3f} s")
        return line, time_text

    # --- 6. Create and Save the Animation ---
    ani = FuncAnimation(
        fig,
        update_animation,
        frames=num_frames,
        blit=False, # Blit must be False for 3D plot text to update correctly
        interval=40 # milliseconds between frames
    )

    try:
        output_filename = 'kirchhoff_rod_replay.gif'
        print(f"\nSaving animation to '{output_filename}'...")
        ani.save(output_filename, writer='pillow', fps=15, dpi=120)
        print(f"Animation saved successfully!")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Please ensure you have 'pillow' installed (`pip install pillow`).")

    # --- 7. Display the Animation ---
    print("Displaying animation...")
    plt.show()


if __name__ == '__main__':
    create_animation_from_files()
