import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
import os

def create_interactive_plotter(history_dir="simulation_history"):
    """
    Loads all simulation data from text files and creates an interactive
    dashboard with linked 3D and 2D plots, controlled by a time slider
    and checkboxes.
    """
    print(f"--- Interactive Kirchhoff Rod Plotter ---")
    print(f"Looking for simulation data in: '{history_dir}'")

    # --- 1. Load All Available History Data ---
    try:
        # Helper to load a single file
        def load_history_file(filename, params):
            path = os.path.join(history_dir, filename)
            if not os.path.exists(path):
                print(f"Warning: History file not found: {filename}. Skipping.")
                return None
            print(f"Loading {filename}...")
            flat_data = np.loadtxt(path, skiprows=1)
            return flat_data.reshape(params['num_frames'], params['M'], 3)

        # Load parameters first to get dimensions
        params_path = os.path.join(history_dir, 'simulation_params.txt')
        if not os.path.exists(params_path):
            print(f"Error: 'simulation_params.txt' not found in '{history_dir}'.")
            return

        params = {}
        with open(params_path, 'r') as f:
            for line in f:
                key, value = line.strip().split(' = ')
                try:
                    if '.' in value or 'e' in value: params[key] = float(value)
                    else: params[key] = int(value)
                except ValueError: params[key] = value
        
        # Load all history files
        history = {
            'X': load_history_file('history_X.txt', params),
            'u': load_history_file('history_u.txt', params),
            'w': load_history_file('history_w.txt', params),
            'f': load_history_file('history_f.txt', params),
            'n': load_history_file('history_n.txt', params),
        }
        # Remove entries that failed to load
        history = {k: v for k, v in history.items() if v is not None}
        if 'X' not in history:
            print("Error: 'history_X.txt' is essential but could not be loaded.")
            return

    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return

    # --- 2. Set up Figure and Axes ---
    # Adjusted figure size and grid spec for 3 plots instead of 4
    fig = plt.figure(figsize=(18, 12)) 
    # Removed one column from gs for the 'starting trajectory' plot
    gs = fig.add_gridspec(2, 3, width_ratios=[7, 6, 4], height_ratios=[20, 1])

    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    
    gs_right = gs[0, 1].subgridspec(2, 1, height_ratios=[2, 1])
    ax2d = fig.add_subplot(gs_right[0])
    ax_check2d = fig.add_subplot(gs_right[1])

    ax_yz = fig.add_subplot(gs[0, 2]) 
    # ax_start_trace subplot removed here

    slider_ax = fig.add_subplot(gs[1, 0])
    ax_check3d = fig.add_subplot(gs[1, 1]) # This ax_check3d will now be in gs[1, 1]

    fig.subplots_adjust(wspace=0.4) 

    # --- 3. Initialize Plots ---
    
    # 3D Plot
    s_vals = np.arange(params['M']) * params['ds']
    initial_X = history['X'][0]
    line3d, = ax3d.plot(initial_X[:, 0], initial_X[:, 1], initial_X[:, 2], 'o-', lw=2.5, color='cornflowerblue')
    
    active_quivers = {}

    all_coords = history['X'].reshape(-1, 3)
    center = np.mean(all_coords, axis=0)
    max_range = np.max(np.ptp(all_coords, axis=0))
    plot_range = max_range * 0.7 if max_range > 0 else 1.0
    ax3d.set_xlim(center[0] - plot_range, center[0] + plot_range)
    ax3d.set_ylim(center[1] - plot_range, center[1] + plot_range)
    ax3d.set_zlim(center[2] - plot_range, center[2] + plot_range)
    ax3d.set_xlabel("X (um)"); ax3d.set_ylabel("Y (um)"); ax3d.set_zlabel("Z (um)")
    ax3d.set_title("3D Rod Configuration")

    # 2D Plot (Arc Length vs. Quantity)
    plot_styles = {
        'u_x': ('-r', 'u'), 'u_y': ('-g', 'u'), 'u_z': ('-b', 'u'),
        'f_x': ('--r', 'f'), 'f_y': ('--g', 'f'), 'f_z': ('--b', 'f'),
        'w_x': ('-.r', 'w'), 'w_y': ('-.g', 'w'), 'w_z': ('-.b', 'w'),
        'n_x': (':r', 'n'), 'n_y': (':g', 'n'), 'n_z': (':b', 'n'),
    }
    
    lines2d = {}
    for label, (style, key) in plot_styles.items():
        if key in history:
            line, = ax2d.plot(s_vals, np.zeros_like(s_vals), style, label=label, visible=False)
            lines2d[label] = line

    ax2d.set_xlabel("Arc Length s (um)")
    ax2d.set_ylabel("Value")
    ax2d.set_title("Physical Quantities along Rod")
    ax2d.grid(True)

    # 2D Y-Z Plane Plot
    # Plot the full Y-Z trajectory of the starting point statically as a faint background
    start_y_coords = history['X'][:, 0, 1]
    start_z_coords = history['X'][:, 0, 2]
    ax_yz.plot(start_y_coords, start_z_coords, 'o-', lw=0.5, color='lightgray', alpha=0.5, label='Start Point Trajectory')

    #line_yz, = ax_yz.plot(initial_X[:, 1], initial_X[:, 2], 'o-', lw=2.5, color='darkorange', label='Current Rod Y-Z')
    ax_yz.set_xlabel("Y (um)")
    ax_yz.set_ylabel("Z (um)")
    ax_yz.set_title("Rod Configuration (Y-Z Plane)")
    ax_yz.grid(True)
    ax_yz.set_aspect('equal', adjustable='box') 
    ax_yz.set_xlim(center[1] - plot_range, center[1] + plot_range)
    ax_yz.set_ylim(center[2] - plot_range, center[2] + plot_range)
    ax_yz.legend() 

    # --- Starting Point Trajectory Plot (X-Y plane) removed ---
    # The following code block has been removed:
    # start_point_history = history['X'][:, 0, :] 
    # ax_start_trace.plot(...)
    # current_start_marker, = ax_start_trace.plot(...)
    # ax_start_trace.plot(...)
    # ax_start_trace.plot(...)
    # ax_start_trace.set_xlabel(...)
    # ax_start_trace.set_ylabel(...)
    # ax_start_trace.set_title(...)
    # ax_start_trace.grid(True)
    # ax_start_trace.legend()
    # trace_x_min, trace_x_max = np.min(start_point_history[:, 0]), np.max(start_point_history[:, 0])
    # trace_y_min, trace_y_max = np.min(start_point_history[:, 1]), np.max(start_point_history[:, 1])
    # x_buffer = max((trace_x_max - trace_x_min) * 0.1, 1e-6) 
    # y_buffer = max((trace_y_max - trace_y_min) * 0.1, 1e-6)
    # ax_start_trace.set_xlim(trace_x_min - x_buffer, trace_x_max + x_buffer)
    # ax_start_trace.set_ylim(trace_y_min - y_buffer, trace_y_max + y_buffer)
    # ax_start_trace.set_aspect('equal', adjustable='box')


    # --- 4. Create Widgets ---

    # Time Slider
    time_slider = Slider(
        ax=slider_ax, label='Time Frame', valmin=0,
        valmax=params['num_frames'] - 1, valinit=0, valstep=1,
    )

    # Checkboxes for 3D overlays
    labels3d = ['Show Forces (f)', 'Show Velocities (u)']
    check3d = CheckButtons(ax_check3d, labels3d, [False, False])

    # Checkboxes for 2D plots
    labels2d = [label for label, (_, key) in plot_styles.items() if key in history]
    check2d = CheckButtons(ax_check2d, labels2d, [False] * len(labels2d))

    # --- 5. Define Update Logic ---
    def update(val):
        frame_idx = int(time_slider.val)
        X_data = history['X'][frame_idx]

        # Update 3D plot line
        line3d.set_data(X_data[:, 0], X_data[:, 1])
        line3d.set_3d_properties(X_data[:, 2])

        # Update 2D YZ plot line
        #line_yz.set_data(X_data[:, 1], X_data[:, 2])

        # --- Update Current Starting Point Marker removed ---
        # current_start_marker.set_data(...) removed from here

        # --- Remove old quivers before drawing new ones ---
        for key in list(active_quivers.keys()):
            active_quivers[key].remove()
            del active_quivers[key]

        # Check status and redraw quivers if needed
        status3d = check3d.get_status()
        quiver_length_scale = plot_range * 0.05 

        if status3d[0] and 'f' in history: # Show Forces
            f_data = history['f'][frame_idx]
            f_magnitudes = np.linalg.norm(f_data, axis=1, keepdims=True)
            f_normalized = np.divide(f_data, f_magnitudes, where=f_magnitudes!=0)
            active_quivers['f'] = ax3d.quiver(
                X_data[:, 0], X_data[:, 1], X_data[:, 2],
                f_normalized[:, 0], f_normalized[:, 1], f_normalized[:, 2],
                length=quiver_length_scale, color='red', label='Forces (f)'
            )

        if status3d[1] and 'u' in history: # Show Velocities
            u_data = history['u'][frame_idx]
            u_magnitudes = np.linalg.norm(u_data, axis=1, keepdims=True)
            u_normalized = np.divide(u_data, u_magnitudes, where=u_magnitudes!=0)
            active_quivers['u'] = ax3d.quiver(
                X_data[:, 0], X_data[:, 1], X_data[:, 2],
                u_normalized[:, 0], u_normalized[:, 1], u_normalized[:, 2],
                length=quiver_length_scale, color='green', label='Velocities (u)'
            )

        # Update 2D plot lines
        status2d = check2d.get_status()
        for i, label in enumerate(labels2d):
            line = lines2d[label]
            if status2d[i]:
                key, component = plot_styles[label][1], label.split('_')[1]
                component_idx = {'x': 0, 'y': 1, 'z': 2}[component]
                line.set_ydata(history[key][frame_idx, :, component_idx])
                line.set_visible(True)
            else:
                line.set_visible(False)
        
        # Rescale 2D plot axes and update legend
        ax2d.relim()
        ax2d.autoscale_view()
        ax2d.legend(loc='upper right', fontsize='small')

        fig.canvas.draw_idle()

    # --- 6. Connect Widgets to Update Function ---
    time_slider.on_changed(update)
    check3d.on_clicked(update)
    check2d.on_clicked(update)

    # --- 7. Initial Call and Display ---
    update(0)
    plt.show()

if __name__ == '__main__':
    create_interactive_plotter()