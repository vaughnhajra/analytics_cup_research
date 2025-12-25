import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, find_peaks

def compute_all_kinematics(df, fps=10, window=7, poly=3, max_jump=5):
    """
    Compute smoothed kinematics (speed, direction, acceleration) for all tracked entities.
    Calculates velocity and acceleration vector components for spatial analysis.
    """
    dt = 1 / fps
    df = df.copy()
    
    # Identify unique player/ball IDs from coordinate columns
    player_ids = sorted({col.split("_")[0] for col in df.columns if col.endswith("_x")})
    results = {}

    for pid in player_ids:
        x_col, y_col = f"{pid}_x", f"{pid}_y"
        z_col = f"{pid}_z"
        if x_col not in df or y_col not in df:
            continue

        # Set physical constraints based on whether the entity is a player or the ball
        is_ball = pid == "ball"
        jump_limit = 40 if is_ball else max_jump
        accel_limit = 50 if is_ball else 8

        x = df[x_col].astype(float).copy()
        y = df[y_col].astype(float).copy()

        def smooth_and_compute_2d(x, y, suffix=""):
            # Identify and remove tracking 'jumps' or teleports that exceed physical limits
            dx = x.diff()
            dy = y.diff()
            dist = np.sqrt(dx**2 + dy**2)
            valid = dist <= jump_limit
            x[~valid] = np.nan
            y[~valid] = np.nan

            # Interpolate missing values and apply Savitzky-Golay smoothing to coordinates
            x = x.interpolate(limit=5, limit_direction="both")
            y = y.interpolate(limit=5, limit_direction="both")

            if len(x.dropna()) < window:
                return

            try:
                dynamic_window = min(window, len(x.dropna()) // 2 * 2 + 1)
                x_smooth = savgol_filter(x, dynamic_window, poly, mode="interp")
                y_smooth = savgol_filter(y, dynamic_window, poly, mode="interp")
            except ValueError:
                return

            # Differentiate smoothed coordinates to obtain velocity components
            vx = np.gradient(x_smooth, dt)
            vy = np.gradient(y_smooth, dt)
            
            # Smooth velocity components to calculate clean horizontal and vertical acceleration
            vx_smooth = savgol_filter(vx, dynamic_window, poly, mode="interp")
            vy_smooth = savgol_filter(vy, dynamic_window, poly, mode="interp")
            
            ax = np.gradient(vx_smooth, dt)
            ay = np.gradient(vy_smooth, dt)

            # Filter and re-interpolate acceleration components to remove outliers
            ax[np.abs(ax) > accel_limit] = np.nan
            ay[np.abs(ay) > accel_limit] = np.nan
            ax = pd.Series(ax).interpolate(limit=5, limit_direction="both").to_numpy()
            ay = pd.Series(ay).interpolate(limit=5, limit_direction="both").to_numpy()

            ax_smooth = savgol_filter(ax, dynamic_window, poly, mode="interp")
            ay_smooth = savgol_filter(ay, dynamic_window, poly, mode="interp")

            # Calculate scalar speed, direction, and smoothed scalar acceleration
            speed = np.sqrt(vx**2 + vy**2)
            direction = np.unwrap(np.arctan2(vy, vx))

            speed_smooth = savgol_filter(speed, dynamic_window, poly, mode="interp")
            accel = np.gradient(speed_smooth, dt)
            accel[np.abs(accel) > accel_limit] = np.nan
            accel = pd.Series(accel).interpolate(limit=5, limit_direction="both").to_numpy()
            accel_smooth = savgol_filter(accel, dynamic_window, poly, mode="interp")

            # Store all derived 2D kinematic features
            results[f"{pid}_x_smooth{suffix}"] = x_smooth
            results[f"{pid}_y_smooth{suffix}"] = y_smooth
            results[f"{pid}_s{suffix}"] = speed_smooth
            results[f"{pid}_d{suffix}"] = direction
            results[f"{pid}_a{suffix}"] = accel_smooth
            results[f"{pid}_ax{suffix}"] = ax_smooth
            results[f"{pid}_ay{suffix}"] = ay_smooth

        # Process standard 2D kinematics for every entity
        smooth_and_compute_2d(x.copy(), y.copy(), suffix="_2d" if pid == "ball" else "")

        # Optional: Process 3D ball kinematics if vertical coordinate (z) is available
        if pid == "ball" and z_col in df:
            z = df[z_col].astype(float).copy()
            dx = x.diff()
            dy = y.diff()
            dz = z.diff()
            dist3d = np.sqrt(dx**2 + dy**2 + dz**2)
            valid = dist3d <= jump_limit
            x[~valid] = np.nan
            y[~valid] = np.nan
            z[~valid] = np.nan

            x = x.interpolate(limit=5, limit_direction="both")
            y = y.interpolate(limit=5, limit_direction="both")
            z = z.interpolate(limit=5, limit_direction="both")

            if len(x.dropna()) < window:
                continue

            try:
                dynamic_window = min(window, len(x.dropna()) // 2 * 2 + 1)
                x_smooth = savgol_filter(x, dynamic_window, poly, mode="interp")
                y_smooth = savgol_filter(y, dynamic_window, poly, mode="interp")
                z_smooth = savgol_filter(z, dynamic_window, poly, mode="interp")
            except ValueError:
                continue

            vx = np.gradient(x_smooth, dt)
            vy = np.gradient(y_smooth, dt)
            vz = np.gradient(z_smooth, dt)
            
            speed3d = np.sqrt(vx**2 + vy**2 + vz**2)
            azimuth = np.unwrap(np.arctan2(vy, vx))
            elevation = np.arctan2(vz, np.sqrt(vx**2 + vy**2))

            speed3d_smooth = savgol_filter(speed3d, dynamic_window, poly, mode="interp")
            accel3d = np.gradient(speed3d_smooth, dt)
            accel3d[np.abs(accel3d) > accel_limit] = np.nan
            accel3d = pd.Series(accel3d).interpolate(limit=5, limit_direction="both").to_numpy()
            accel3d_smooth = savgol_filter(accel3d, dynamic_window, poly, mode="interp")

            results[f"{pid}_x_smooth_3d"] = x_smooth
            results[f"{pid}_y_smooth_3d"] = y_smooth
            results[f"{pid}_z_smooth_3d"] = z_smooth
            results[f"{pid}_s_3d"] = speed3d_smooth
            results[f"{pid}_d_azimuth_3d"] = azimuth
            results[f"{pid}_d_elev_3d"] = elevation
            results[f"{pid}_a_3d"] = accel3d_smooth

    # Consolidate results into a DataFrame and merge with the original tracking data
    results_df = pd.DataFrame(results, index=df.index)
    df_new = pd.concat([df, results_df], axis=1)
    return df_new

def calculate_player_distance(df_kinematics, player_id, start_frame, end_frame):
    
    """
    Calculates the total distance traveled by a player between a start and end frame.
    (This function is defined outside of add_model_features for cleanliness)
    """
    
    x_col = f'{player_id}_x_smooth'
    y_col = f'{player_id}_y_smooth'

    if x_col not in df_kinematics.columns or y_col not in df_kinematics.columns:
        return 0.0

    kinematics_slice = df_kinematics.loc[
        (df_kinematics.index >= start_frame) & 
        (df_kinematics.index <= end_frame), 
        [x_col, y_col]
    ].copy()
    
    kinematics_slice.dropna(inplace=True)
    
    if len(kinematics_slice) < 2:
        return 0.0
    
    dx = kinematics_slice[x_col] - kinematics_slice[x_col].shift(1)
    dy = kinematics_slice[y_col] - kinematics_slice[y_col].shift(1)
    
    distances = np.sqrt(dx**2 + dy**2)
    
    total_distance = distances.sum()
    
    return total_distance if pd.notna(total_distance) else 0.0

def find_passes_by_events(
    df,
    x_col='ball_x_smooth_2d',
    y_col='ball_y_smooth_2d',
    speed_col='ball_s_2d',
    accel_col='ball_a_2d',
    team_id_col='ball_owning_team_id',
    fps=10,
    kick_thresh=15.0,
    receive_thresh=10.0,
    peak_distance_s=0.2,
    min_pass_duration_s=0.3,
    max_pass_duration_s=4.0,
    min_avg_pass_speed=5.0
):
    """
    Detects ball passes by identifying a "kick" event (high acceleration peak)
    followed by a "receive" event (high deceleration peak).

    Returns a DataFrame of validated pass events.
    """
    
    df = df.copy()

    # Extract Data to NumPy Arrays
    accel_data = df[accel_col].to_numpy()
    speed_data = df[speed_col].to_numpy()
    team_id_data = df[team_id_col].to_numpy()
    x_data = df[x_col].to_numpy() if x_col in df else None
    y_data = df[y_col].to_numpy() if y_col in df else None

    # Handle NaNs in acceleration data, as find_peaks requires finite values
    accel_data_nonan = np.nan_to_num(accel_data, nan=0.0)

    # Convert Time-based Params to Frames
    peak_distance_frames = int(peak_distance_s * fps)
    min_pass_duration_frames = int(min_pass_duration_s * fps)
    max_pass_duration_frames = int(max_pass_duration_s * fps)

    # Find Kick & Receive Events (Peaks)

    # Find all sharp acceleration (kick) peaks
    kick_indices, _ = find_peaks(
        accel_data_nonan,
        height=kick_thresh,
        distance=peak_distance_frames
    )

    # Find all sharp deceleration (receive) peaks by finding peaks on the *negative* acceleration
    receive_indices, _ = find_peaks(
        -accel_data_nonan, # Note the negative sign
        height=receive_thresh,
        distance=peak_distance_frames
    )

    # Filter kicks: A valid pass must be initiated by a player in possession.
    valid_kick_indices = []
    for idx in kick_indices:
        if pd.notna(team_id_data[idx]):
            valid_kick_indices.append(idx)
    kick_indices = np.array(valid_kick_indices)

    if len(kick_indices) == 0 or len(receive_indices) == 0:
        # No passes possible if no kicks or receives were found
        return pd.DataFrame()

    # Pair Kicks to Receives & Validate

    valid_passes = []

    # Use searchsorted to efficiently find the next receive for each kick
    potential_receive_locs = np.searchsorted(receive_indices, kick_indices, side='right')

    for i, kick_idx in enumerate(kick_indices):
        
        if potential_receive_locs[i] >= len(receive_indices):
            # This kick is after the last receive, so it has no pair
            continue

        receive_idx = receive_indices[potential_receive_locs[i]]

        # Validation 1: Check for Interruptions
        # If there is another kick before this receive, this pass is invalid
        if (i + 1) < len(kick_indices) and kick_indices[i+1] < receive_idx:
            continue

        # Validation 2: Check Pass Duration
        duration_frames = receive_idx - kick_idx
        if not (min_pass_duration_frames <= duration_frames <= max_pass_duration_frames):
            continue

        # Validation 3: Check Pass Speed
        # The pass "flight" is the period between the kick and receive
        pass_slice = speed_data[kick_idx + 1 : receive_idx]
        if len(pass_slice) == 0:
            continue # Kick and receive are adjacent frames, not a pass

        avg_speed = np.nanmean(pass_slice)
        if avg_speed < min_avg_pass_speed:
            continue # Too slow, likely a dribble or tap

        # If all checks pass, record the event
        kicker_team = team_id_data[kick_idx]
        receiver_team = team_id_data[receive_idx]

        pass_event = {
            "frame_start": kick_idx,
            "frame_end": receive_idx,
            "duration_s": duration_frames / fps,
            "kicker_team_id": kicker_team,
            "receiver_team_id": receiver_team,
            "avg_speed": avg_speed,
            "max_speed": np.nanmax(pass_slice),
            "kick_accel": accel_data[kick_idx],
            "receive_decel": accel_data[receive_idx], # This will be negative
            "start_x": x_data[kick_idx] if x_data is not None else np.nan,
            "start_y": y_data[kick_idx] if y_data is not None else np.nan,
            "end_x": x_data[receive_idx] if x_data is not None else np.nan,
            "end_y": y_data[receive_idx] if y_data is not None else np.nan,
        }
        valid_passes.append(pass_event)

    return pd.DataFrame(valid_passes)