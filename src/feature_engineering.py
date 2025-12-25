import pandas as pd
import numpy as np
from .kinematics import calculate_player_distance
from src.kinematics import compute_all_kinematics, find_passes_by_events
from src.data_loader import load_match_data

def compute_distance_to_ball(df, player_id):
    """Euclidean 2D distance from player to ball using smoothed coordinates."""
    bx, by = df["ball_x_smooth_2d"], df["ball_y_smooth_2d"]
    px, py = df[f"{player_id}_x_smooth"], df[f"{player_id}_y_smooth"]
    return np.sqrt((px - bx)**2 + (py - by)**2)

def compute_angle_to_ball(df, player_id):
    """Relative angle between player's movement direction and direction to ball."""
    px_col = f"{player_id}_x_smooth"
    py_col = f"{player_id}_y_smooth"
    dir_col = f"{player_id}_d"

    if any(col not in df.columns for col in [px_col, py_col, dir_col]):
        return np.full(len(df), np.nan)

    px = df[px_col].to_numpy()
    py = df[py_col].to_numpy()
    bx = df["ball_x_smooth_2d"].to_numpy()
    by = df["ball_y_smooth_2d"].to_numpy()
    move_dir = pd.to_numeric(df[dir_col], errors="coerce").to_numpy()  # ensures numeric

    if np.all(np.isnan(move_dir)):
        return np.full(len(df), np.nan)

    theta_to_ball = np.arctan2(by - py, bx - px)
    move_dir_mod = (move_dir + np.pi) % (2 * np.pi) - np.pi
    angle_diff = theta_to_ball - move_dir_mod
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    return angle_diff

def compute_total_distance(df, player_id):
    """Cumulative distance traveled by player using smoothed positions."""
    px, py = df[f"{player_id}_x_smooth"], df[f"{player_id}_y_smooth"]
    dx, dy = px.diff(), py.diff()
    return np.nancumsum(np.sqrt(dx**2 + dy**2))

def add_player_ball_features(df):
    # Remove any duplicate columns (keeps last)
    df = df.loc[:, ~df.columns.duplicated(keep='last')].copy()

    player_ids = [c.split("_x_smooth")[0] for c in df.columns if c.endswith("_x_smooth") and "ball" not in c]

    # Call helper functions to compute features
    for pid in player_ids:
        df[f"{pid}_dist_to_ball"] = compute_distance_to_ball(df, pid)
        df[f"{pid}_angle_to_ball"] = compute_angle_to_ball(df, pid)
        df[f"{pid}_total_dist"] = compute_total_distance(df, pid)

    return df

def detect_player_responses(
    df_kinematics, 
    passes, 
    player_team_map,
    fps=10, 
    reaction_window_s=1.25, 
    reaction_thresh_ms2=1.5,
    return_stats=False,
    max_distance=50.0
):
    """
    Detects player reactions to passes based on sudden changes in their 2D acceleration vector.
    Processes defenders within a specific distance threshold from the ball at the moment of the pass.
    """
    
    all_responses = []
    player_stats = {}

    # Extract unique player IDs from acceleration columns, excluding ball data
    player_ids = sorted({
        col.split("_")[0] for col in df_kinematics.columns 
        if col.endswith("_ax") and not col.startswith("ball")
    })
    
    # Initialize response tracking for all identified players
    for pid in player_ids:
        player_stats[pid] = {'possible': 0, 'actual': 0}
    
    if not player_ids:
        return pd.DataFrame() if not return_stats else (pd.DataFrame(), pd.DataFrame())

    reaction_window_frames = int(reaction_window_s * fps)
    n_frames = len(df_kinematics)

    # Pre-convert kinematic columns to NumPy arrays for vectorized performance
    player_data = {}
    for pid in player_ids:
        px = df_kinematics.get(f"{pid}_x", pd.Series(np.nan)).to_numpy()
        py = df_kinematics.get(f"{pid}_y", pd.Series(np.nan)).to_numpy()
        
        player_data[pid] = {
            "ax": df_kinematics[f"{pid}_ax"].to_numpy(),
            "ay": df_kinematics[f"{pid}_ay"].to_numpy(),
            "x": px,
            "y": py
        }

    # Extract ball positional data
    ball_x = df_kinematics.get("ball_x", pd.Series(np.nan)).to_numpy()
    ball_y = df_kinematics.get("ball_y", pd.Series(np.nan)).to_numpy()

    # Process each pass event
    for pass_index, pass_row in passes.iterrows():
        pass_start_frame = int(pass_row['frame_start'])
        
        # Define the temporal window to scan for a reaction
        window_start = pass_start_frame + 1 
        window_end = min(pass_start_frame + reaction_window_frames, n_frames)

        if window_start >= window_end:
            continue 

        # Determine ball location at the start of the pass
        bx_at_pass = ball_x[pass_start_frame] if len(ball_x) > pass_start_frame else np.nan
        by_at_pass = ball_y[pass_start_frame] if len(ball_y) > pass_start_frame else np.nan

        for pid in player_ids:
            p_dict = player_data[pid]
            ax_data = p_dict["ax"]
            ay_data = p_dict["ay"]
            
            # Establish baseline acceleration at the frame the pass is initiated
            ax_base = ax_data[pass_start_frame]
            ay_base = ay_data[pass_start_frame]

            if pd.isna(ax_base) or pd.isna(ay_base):
                continue 

            is_within_range = False
            kicker_team = pass_row['kicker_team_id']
            player_team = player_team_map.get(pid)
            
            # Logic Filter: Only analyze defenders (non-kicking team) within the distance threshold
            if player_team != kicker_team:
                px_at_pass = p_dict["x"][pass_start_frame]
                py_at_pass = p_dict["y"][pass_start_frame]
        
                if not pd.isna(bx_at_pass) and not pd.isna(px_at_pass):
                    dist = np.sqrt((px_at_pass - bx_at_pass)**2 + (py_at_pass - by_at_pass)**2)
                    
                    if dist <= max_distance:
                        is_within_range = True
                        player_stats[pid]['possible'] += 1
    
            # Skip reaction calculation for players outside the contextual filter
            if not is_within_range:
                continue
    
            # Calculate change in acceleration vector magnitude across the window
            ax_window = ax_data[window_start:window_end]
            ay_window = ay_data[window_start:window_end]

            ax_delta = ax_window - ax_base
            ay_delta = ay_window - ay_base
            delta_magnitude = np.sqrt(ax_delta**2 + ay_delta**2)
            
            # Find the first frame where change in acceleration exceeds the threshold
            reaction_frames_relative = np.where(delta_magnitude > reaction_thresh_ms2)[0]

            if len(reaction_frames_relative) > 0:
                first_reaction_relative_idx = reaction_frames_relative[0]
                response_frame_absolute = window_start + first_reaction_relative_idx
                
                reaction_time_s = (response_frame_absolute - pass_start_frame) / fps
                reaction_magnitude = delta_magnitude[first_reaction_relative_idx]

                # Store individual reaction event details
                all_responses.append({
                    "pass_index": pass_index,
                    "player_id": pid,
                    "response_frame": response_frame_absolute,
                    "reaction_time_s": reaction_time_s,
                    "reaction_magnitude_ms2": reaction_magnitude,
                    "pass_start_frame": pass_start_frame,
                    "base_ax": ax_base,
                    "base_ay": ay_base,
                })

                if is_within_range:
                    player_stats[pid]['actual'] += 1

    df_responses = pd.DataFrame(all_responses)

    if not return_stats:
        return df_responses

    # Consolidate player statistics into a summary frequency table
    summary_data = []
    for pid, counts in player_stats.items():
        possible = counts['possible']
        actual = counts['actual']
        freq = (actual / possible) if possible > 0 else 0.0
        
        summary_data.append({
            "player_id": pid,
            "possible_responses": possible,
            "number_of_responses": actual,
            "response_frequency": freq
        })
    
    df_stats = pd.DataFrame(summary_data)
    df_stats = df_stats.sort_values(by="response_frequency", ascending=False).reset_index(drop=True)

    return df_responses, df_stats

def add_model_features(
    df_reactions,
    df_kinematics,
    df_passes,
    meta_info,
    de_match,
    remove_attacking_team=True,
    max_reaction_time_s=1.25,
    max_distance_to_passer=50.0,
    fps=10
):
    """
    Enriches the player reactions DataFrame with spatial, tactical, and contextual features.
    Includes player load (distance traveled), score differentials, peer reaction times, 
    and detailed pitch-geometry based threat and perception metrics.
    """
    
    model_df = df_reactions.copy()
    
    # Filter rows by maximum reaction time threshold to remove outliers before processing
    model_df = model_df[model_df['reaction_time_s'] <= max_reaction_time_s].copy()
    if model_df.empty:
        return pd.DataFrame()

    # Calculate peer context by determining the average reaction time of all other defenders on the same pass
    pass_group = model_df.groupby('pass_index')['reaction_time_s']
    pass_mean_rt = pass_group.transform('mean')
    pass_count = pass_group.transform('count')
    
    model_df['other_players_mean_rt'] = np.where(
        pass_count > 1,
        (pass_count * pass_mean_rt - model_df['reaction_time_s']) / (pass_count - 1),
        np.nan
    )
    
    # Map teams and players to determine defensive responsibilities and pitch directions
    home_team_id = meta_info["teams"][0]["team_id"]
    away_team_id = meta_info["teams"][1]["team_id"]
    home_players = [p["player_id"] for p in meta_info["teams"][0]["players"]]
    away_players = [p["player_id"] for p in meta_info["teams"][1]["players"]]
    all_player_ids = home_players + away_players
    
    player_team_map = {pid: home_team_id for pid in home_players}
    player_team_map.update({pid: away_team_id for pid in away_players})

    pitch_dim = meta_info.get('pitch_dimensions', {'length': 105.0, 'width': 68.0})
    goal_line_x = pitch_dim['length'] / 2.0
    home_goal_x, away_goal_x = -goal_line_x, goal_line_x

    # Join pass physics and spatial metadata from the pass detection results
    pass_cols = ['kicker_team_id', 'avg_speed', 'kick_accel', 'duration_s', 'start_x', 'start_y', 'end_x', 'end_y']
    model_df = model_df.merge(df_passes[pass_cols], left_on='pass_index', right_index=True, how='left')
    model_df.rename(columns={
        'avg_speed': 'pass_avg_speed', 'kick_accel': 'pass_kick_accel', 'duration_s': 'pass_duration_s',
        'start_x': 'pass_start_x', 'start_y': 'pass_start_y', 'end_x': 'pass_end_x', 'end_y': 'pass_end_y'
    }, inplace=True)

    # Capture the kinematic state of the reacting player exactly when the pass was initiated
    lookup = model_df[['player_id', 'pass_start_frame']].drop_duplicates()
    t0_features = []
    for _, row in lookup.iterrows():
        pid, t0 = row['player_id'], int(row['pass_start_frame'])
        frame = df_kinematics.loc[t0] if t0 in df_kinematics.index else {}
        t0_features.append({
            'player_id': pid, 'pass_start_frame': t0,
            'player_x_t0': frame.get(f'{pid}_x_smooth'), 'player_y_t0': frame.get(f'{pid}_y_smooth'),
            'player_s_t0': frame.get(f'{pid}_s'), 'player_d_t0': frame.get(f'{pid}_d')
        })
    model_df = model_df.merge(pd.DataFrame(t0_features), on=['player_id', 'pass_start_frame'], how='left')

    # Calculate physical load metrics including distance covered in the prior 5 minutes and total game distance
    five_min_lookback = 5 * 60 * fps 
    dist_cache = {} 

    def calculate_load(row, mode):
        pid, t0 = row['player_id'], int(row['pass_start_frame'])
        start = 0 if mode == 'game' else max(0, t0 - five_min_lookback)
        key = (pid, start, t0)
        if key not in dist_cache:
            from src.kinematics import calculate_player_distance
            dist_cache[key] = calculate_player_distance(df_kinematics, pid, start, t0)
        return dist_cache[key]

    model_df['distance_last_5_min'] = model_df.apply(calculate_load, axis=1, mode='recent')
    model_df['distance_game_total'] = model_df.apply(calculate_load, axis=1, mode='game')
    
    # Compute basic relative distances between the player, the passer, and the pass destination
    model_df['pass_length'] = np.sqrt((model_df['pass_end_x']-model_df['pass_start_x'])**2 + (model_df['pass_end_y']-model_df['pass_start_y'])**2)
    model_df['distance_to_passer'] = np.sqrt((model_df['player_x_t0']-model_df['pass_start_x'])**2 + (model_df['player_y_t0']-model_df['pass_start_y'])**2)
    model_df['distance_to_pass_end'] = np.sqrt((model_df['player_x_t0']-model_df['pass_end_x'])**2 + (model_df['player_y_t0']-model_df['pass_end_y'])**2)
    model_df['game_time_elapsed_s'] = model_df['pass_start_frame'] / fps
    model_df['player_team'] = model_df['player_id'].map(player_team_map)
    model_df['is_defending_team'] = (model_df['player_team'] != model_df['kicker_team_id']).astype(int)

    # Determine match context by calculating the live score differential relative to the reacting player
    score_timeline = de_match[['frame_start', 'team_id', 'team_score', 'opponent_team_score']].sort_values('frame_start').drop_duplicates('frame_start')
    model_df = pd.merge_asof(model_df.sort_values('pass_start_frame'), score_timeline, left_on='pass_start_frame', right_on='frame_start')
    
    rel_score = np.where(model_df['player_team'] == model_df['team_id'], 
                         model_df['team_score'] - model_df['opponent_team_score'], 
                         model_df['opponent_team_score'] - model_df['team_score'])
    model_df['score_differential'] = rel_score
    model_df['is_winning'], model_df['is_losing'], model_df['is_tied'] = (rel_score > 0).astype(int), (rel_score < 0).astype(int), (rel_score == 0).astype(int)
    model_df.drop(columns=['frame_start', 'team_id', 'team_score', 'opponent_team_score'], inplace=True)
    
    # Calculate social and congestion features including defensive responsibility and proximity to receivers
    congestion_data = []
    for pass_idx, group in model_df.groupby('pass_index'):
        t0 = int(group['pass_start_frame'].iloc[0])
        pass_end = df_passes.loc[pass_idx, ['end_x', 'end_y']].values
        if t0 not in df_kinematics.index: continue
        
        frame = df_kinematics.loc[t0]
        pos = {pid: (frame[f'{pid}_x_smooth'], frame[f'{pid}_y_smooth']) for pid in all_player_ids if pd.notna(frame.get(f'{pid}_x_smooth'))}

        for _, react in group.iterrows():
            pid = react['player_id']
            if pid not in pos: continue
            
            p_pos = np.array(pos[pid])
            team = player_team_map[pid]
            teammates = [p for p, t in player_team_map.items() if t == team and p != pid and p in pos]
            opponents = [p for p, t in player_team_map.items() if t != team and p in pos]
            
            opp_dists = [np.linalg.norm(p_pos - pos[o]) for o in opponents]
            opp_to_end = [np.linalg.norm(pass_end - pos[o]) for o in opponents]
            tm_to_end = [np.linalg.norm(pass_end - pos[t]) for t in teammates]
            
            congestion_data.append({
                'pass_index': pass_idx, 'player_id': pid,
                'opponents_within_5m_t0': sum(1 for d in opp_dists if d <= 5),
                'distance_to_proximal_receiver': min(opp_to_end) if opp_to_end else np.nan,
                'is_closest_defender_to_pass_end': int(react['distance_to_pass_end'] <= (min(tm_to_end) if tm_to_end else np.inf)),
                'is_last_defender': int(all(p_pos[0] <= pos[t][0] if team == home_team_id else p_pos[0] >= pos[t][0] for t in teammates)) if teammates else 0
            })
    
    if congestion_data:
        model_df = model_df.merge(pd.DataFrame(congestion_data), on=['pass_index', 'player_id'], how='left')

    # Add high-level threat metrics and perception features such as player orientation relative to the passer
    goal_x = np.where(model_df['player_team'] == home_team_id, home_goal_x, away_goal_x)
    model_df['distance_pass_end_to_goal'] = np.sqrt((model_df['pass_end_x'] - goal_x)**2 + model_df['pass_end_y']**2)
    
    v_dist = np.where(model_df['kicker_team_id'] == home_team_id, model_df['pass_end_x'] - model_df['pass_start_x'], model_df['pass_start_x'] - model_df['pass_end_x'])
    model_df['pass_verticality'], model_df['pass_progressive_distance'] = v_dist, np.maximum(0, v_dist)
    
    dx, dy = model_df['pass_start_x'] - model_df['player_x_t0'], model_df['pass_start_y'] - model_df['player_y_t0']
    angle_to_passer = np.abs(np.degrees(np.arctan2(dy, dx) - model_df['player_d_t0']))
    model_df['player_facing_passer_angle'] = np.minimum(angle_to_passer, 360 - angle_to_passer)

    # Perform final spatial filtering and remove any remaining observations with missing values
    model_df = model_df[model_df['distance_to_passer'] <= max_distance_to_passer]
    if remove_attacking_team:
        model_df = model_df[model_df['is_defending_team'] == 1]

    cols_to_keep = [
        'reaction_time_s', 'player_s_t0', 'distance_last_5_min', 'distance_game_total',
        'pass_avg_speed', 'pass_length', 'distance_to_passer', 'game_time_elapsed_s',
        'score_differential', 'other_players_mean_rt', 'distance_pass_end_to_goal',
        'pass_progressive_distance', 'player_facing_passer_angle', 'opponents_within_5m_t0',
        'distance_to_proximal_receiver', 'is_last_defender', 'pass_index', 'player_id'
    ]
    
    model_df = model_df[[c for c in cols_to_keep if c in model_df.columns]].dropna()
    
    return model_df


def process_match_pipeline(match_id, loader_func, kinematics_func, feature_func):
    """
    Consolidates data loading, kinematics, and feature engineering for a single match.
    Returns a model-ready DataFrame and the player response frequency statistics.
    """
    
    # Load Data
    tracking_df, events_df, match_info = load_match_data(match_id)

    # Create Map of Teams
    home_team_id = match_info["teams"][0]["team_id"]
    away_team_id = match_info["teams"][1]["team_id"]
    
    player_team_map = {}
    for p in match_info["teams"][0]["players"]:
        player_team_map[p["player_id"]] = home_team_id
    for p in match_info["teams"][1]["players"]:
        player_team_map[p["player_id"]] = away_team_id

    # Compute smoothed player + ball kinematics (using your notebook settings)
    df_new = compute_all_kinematics(tracking_df, fps=10, window=3, poly=2, max_jump=5)

    # Add player-ball features and detect passes (both functions within this file)
    df_new_ball = add_player_ball_features(df_new)
    passes = find_passes_by_events(df_new_ball, fps=10)
    
    # Detect player responses
    responses, df_percentage = detect_player_responses(
        df_new_ball, 
        passes, 
        player_team_map=player_team_map, 
        reaction_thresh_ms2=3.5, 
        return_stats=True, 
        max_distance=50
    )

    # Create the Model-Ready Feature Dataset (add_model_features function within this file)
    model_df_single_match = add_model_features(
        responses,                     
        df_new_ball,              
        passes,                   
        match_info,                
        events_df,
        remove_attacking_team=True, 
        max_reaction_time_s=1.25,   
        max_distance_to_passer=50.0, 
        fps=10
    )

    model_df_single_match['match_id'] = match_id
    
    return model_df_single_match, df_percentage