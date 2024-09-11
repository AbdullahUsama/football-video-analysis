# Football Video Analysis

This project aims to analyze football match footage, extracting insights such as player movement, ball tracking, and camera motion estimation.

## Features

- **Camera Movement Estimator**: Tracks camera motion for smoother analysis.
- **Player and Ball Assigner**: Detects players and tracks ball possession.
- **Distance Calculator**: Measures player distance.
- **Team Assigner**: Categorizes players by teams.
- **View Transformer**: Maps tilted camera angle onto a square to measure correct distance

## Prerequisites

- Python 
- OpenCV
- NumPy
- Additional dependencies listed in `requirements.txt`

## Usage

1. Place input videos in `input_videos/`.
2. Run:

```bash```
python main.py


  Outputs are saved in the `output/` directory.

## Folder Structure

- `camera_movement_estimator/`: Code for camera motion tracking.
- `player_ball_assigner/`: Logic for player and ball detection.
- `team_assigner/`: Player team classification.
- `view_transformer/`: Field view transformation.
- `tracker/`: Player movement tracking.
