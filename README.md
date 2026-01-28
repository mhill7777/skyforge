# SkyForge Fiducial Tracking Experimentation (Python)
> This project was created to learn how to track fiducials, specifically ArUco fiducials. This project does this primarily with a [camera name] and also with [real sense camera] with the Open CV library: opencv-python. It is able to track the relative position, which includes x, y, and z (depth); and rotation around each local axis of the ArUco fiducial. It can also provide the position of a fiducial relative to another one.

## Table of Contents
- [Dependecies](#dependencies)
- [Usage](#usage)
- [Features](#features)
- [Contributors](#contributors)

## Dependencies
This project primarily uses the python libraries:
- `opencv-python`
- `matplotlib`
- `numpy`
- `pandas`

For tracking with the real sense camera, use `[realsense library download command]`

## Usage
>This project can generate ArUco fiducials, generate ArUco calibration boards, calibrate cameras, and track fiducials in 3D space with a basic cameras and real sense camera. Please note that at the time of reading this, some of the files may be irrelevant and were only used for experimentation.

### Relevant files
- **Helper Files**
    - `generateFiducial.py`
    - `generateBoard.py`
    - `calibration.py`
    - `takingPhotos.py`
    - `easyGraphing.py`
        - (only used in 3`dTracking+Relative.py`)
- **Main Files**
    - `3dTracking.py`
    - `3dTracking+Relative.py`
    - `realSenseTracking.py`