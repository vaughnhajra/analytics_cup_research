import pandas as pd
from kloppy import skillcorner

def extract_metadata_info(metadata):
    """
    Extracts and processes match metadata into a structured dictionary.
    args:
        metadata (kloppy.metadata.base.Metadata): The metadata object from Kloppy.
    returns:
        info: A dictionary containing processed match metadata.
    """
    info = {}

    # Basic match info
    info["provider"] = metadata.provider.value
    info["game_id"] = metadata.game_id
    info["date"] = metadata.date.isoformat()
    info["frame_rate"] = metadata.frame_rate
    info["pitch_dimensions_meters"] = {
        "length": metadata.pitch_dimensions.pitch_length,
        "width": metadata.pitch_dimensions.pitch_width,
    }
    info["score"] = {
        "home": metadata.score.home,
        "away": metadata.score.away,
    }

    # Periods
    info["periods"] = [
        {
            "id": p.id,
            "start_seconds": p.start_timestamp.total_seconds(),
            "end_seconds": p.end_timestamp.total_seconds(),
            "duration_seconds": p.end_timestamp.total_seconds() - p.start_timestamp.total_seconds(),
        }
        for p in metadata.periods
    ]

    # Teams and players
    teams = []
    for team in metadata.teams:
        team_info = {
            "team_id": team.team_id,
            "team_name": team.name,
            "ground": team.ground.value,
            "players": [
                {
                    "player_id": p.player_id,
                    "name": p.name,
                    "jersey_no": p.jersey_no,
                    "starting": p.starting,
                    "starting_position": (
                        p.starting_position.name if p.starting_position else None
                    ),
                }
                for p in team.players
            ],
        }
        teams.append(team_info)
    info["teams"] = teams

    return info


def load_match_data(match_id):
    """
    Loads all data (tracking, metadata, events) for a given match ID
    from the SkillCorner open data repository.
    
    Args:
        match_id (int): The ID of the match to load.
        
    Returns:
        tuple: (df, de_match, metadata_info)
            - df (pd.DataFrame): Transformed tracking data (1st half, static orientation).
            - de_match (pd.DataFrame): Dynamic events data.
            - metadata_info (dict): Processed match metadata.
    """

    print(f"Loading & processing match {match_id}: ", end="", flush=True)

    # Define URLs dynamically using the match_id
    tracking_data_github_url = f"https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl"
    meta_data_github_url = f"https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_match.json"
    events_csv_url = f"https://raw.githubusercontent.com/SkillCorner/opendata/refs/heads/master/data/matches/{match_id}/{match_id}_dynamic_events.csv"

    # Load SkillCorner dataset (tracking + metadata)
    try:
        dataset = skillcorner.load(
            meta_data=meta_data_github_url,
            raw_data=tracking_data_github_url,
            coordinates="skillcorner"
        )
    except Exception as e:
        print(f"Error loading SkillCorner data for match {match_id}: {e}")
        print("Please check if the match ID is correct and data is available.")
        return None, None, None

    # Transform tracking data to DataFrame
    df = (
        dataset.transform(
            to_orientation="STATIC_HOME_AWAY"
        )
        .to_df(
            engine="pandas"
        )  # Convert to a Pandas DataFrame
    )

    # Load dynamic events
    try:
        de_match = pd.read_csv(events_csv_url, low_memory=False)
    except Exception as e:
        print(f"Error loading dynamic events CSV for match {match_id}: {e}")
        de_match = pd.DataFrame()  # Return empty DataFrame if loading fails

    # Process metadata using the helper function
    metadata_info = extract_metadata_info(dataset.metadata)
    
    print(f"Success")

    return df, de_match, metadata_info