# SkyForge Fiducial Tracking Experimentation (Python)
This project was created to learn how to track fiducials, specifically ArUco fiducials with the Open CV in python. For this project any basic camera can be used, but for the depth sensing camera, we specifically programmed it for the Intel RealSense D435 Webcam. This project is able to track the relative position to the camera, which includes x, y, and z (depth); and rotation around each local axis of the ArUco fiducial. It can also provide the position of a fiducial relative to another fiducial. This project is made up of many files that run independently of each other and serve different functions.

## Getting Started

### Dependencies
This project primarily uses the python libraries:
- `opencv-python`
- `matplotlib`
- `numpy`
- `pandas`

For tracking with the real sense camera, use `[realsense library download command]`

### Executing Program
Simply run one of the Python files. There shouldn't be any problems as long as you have all of the dependencies and don't have any configuration issues. Common configuration issues come from difficulties with setting the [camera index](#cv2videocapture0) and making sure relative paths are relative to directory where the python code is being ran.

## General Configuration

#### `cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)`
Changing `DICT_6X6_250` in the above line of code changes the dictionary that the unique ArUco markers that can be used. The `6X6` in the middle specifies the grid size and the `250` at the end specifies the size of the dictionary and the number of unique markers that can be used. Either of these parts can be modified to one of the other predefined dictionaries.

#### `cv2.aruco.CharucoBoard(size, squareLength, markerLength, dictionary)`
Configures the ChArUco board that will be used for calibration.
- `size` is a tuple of the number of squares long and high the board will be. Ex. `(9,6)`
- `squareLength` is the size of the squares in meters
- `markerLength` is the size of the fiducial markers in meters (needs to be smaller than `squareLength` in order to have enough whitespace around the marker for easy detection when calibrating cameras)
- `dictionary` specifies the fiducial dictionary ([configuration details](#cv2arucogetpredefineddictionarycv2arucodict_6x6_250))

#### `cv2.VideoCapture(0)`
This specifies the sources of images/frames that will be fed into the program
- The parameter is an index ranging from 0-2 that correspond to different cameras connected to the computer. Experimenting is the easiest way to determine the desired index. If you are having trouble accessing a camera, first ensure the proper permissions are set in settings; if you still have trouble connecting to a camera, run VS code as administrator which can be done by opening VS code from the terminal with the `sudo` command.



## Usage and Specific Configuration
>This project can generate ArUco fiducials markers, generate ArUco calibration boards, calibrate cameras, and track fiducials in 3D space with a basic cameras and real sense camera. Please note that at the time of reading this, some of the files may be irrelevant and were only used for experimentation.

### Helper Files

#### [generateFiducial.py](generateFiducial.py)
Running this file generates a PNG of a fiducial marker with the marker's id in the file name. After running, the PNG will be saved to the local directory and preview of the picture with its id in the title will be displayed.
- `aruco_dict` specifies the marker's dictionary [configuration details](#cv2arucogetpredefineddictionarycv2arucodict_6x6_250)
- `marker_id` specifies the id of the generated fiducial marker and will also be included in the name of the PNG file.
- `marker_size` specifies the marker's resolution and its value is the number of pixels.

#### [generateBoard.py](generateBoard.py)
Generates a PNG of a ChArUco board for camera calibration saved with the name "my_charuco_board.png".
 - `dictionary` specifies the dictionary that will be used to generate the markers [configuration details](#cv2arucogetpredefineddictionarycv2arucodict_6x6_250)
 - `board` specifies the ChArUco board that will be outputted [configuration details](#cv2arucocharucoboardsize-squarelength-markerlength-dictionary) (**Note:** configuration of the board being used for calibration **MUST MATCH** the configuration in code)
 - `pixelsLong` and `pixelsTall` specify the resolution and dimensions of the image (if the dimension are not proportional to the board, the extra space will be white)
 
#### [calibration.py](calibration.py)
Runs through all of the calibration images to outputs a file called "calibration_data.npz" if successful containing the data describing the intrinsic and extrinsic parameters of the camera. In the console, the program will output information about whether an image is accepted or rejected.
- `CHARUCO_BOARD` specifies the ChArUco board used for calibration [configuration details](#cv2arucocharucoboardsize-squarelength-markerlength-dictionary) (**Note:** configuration of the board being used for calibration **MUST MATCH** the configuration in code)
- `fileName` specifies the name of the file that contains the calibration data. Do not include the file extnension. The name of the final output will automatically end in ".npz"
- `image_folder` is the path to the directory containing all of the calibration images. By default it's set to `"../CalibrationImages/"`.

#### [takingPhotos.py](takingPhotos.py)
This program makes it easier to take calibration photos, particularly on Mac. Running this program brings up a window displaying a live feed from a desired camera. Pressing `SPACE` will capture a frame and save it as a PNG in a specified folder with a name containing the number of photos already taken while the program has been running. Pressing `ESC` will close the window and terminate the program.
- `cam` specifies which camera is the source of the video [configuration details](#cv2videocapture0)
- `save_directory` specifies the path to the folder where the captured images will be saved. For ease of use this should match the path that the calibration file draws from.
- `imageNum` is itterated by 1 each time a image is captured. It is included in the PNG name to make the names unique. Changing the value from zero at the beginning of the code will change the name of the first photo and is help when added more images to a file between different runs.

#### Other
- [easyGraphing.py](easyGraphing.py) simply makes generating plots for data with respect to frame index for "3dTracking+Relative.py" more concise.
- [cameraTesting.py](cameraTesting.py) displays all detections in a live camera feed including detections that are rejected.
- [detectImage.py](detectImage.py) takes a single image, draws the detection on that image in a window, and prints a list of detected marker ids.

### Main Files

#### [3dTracking.py](3dTracking.py)
This program opens up a window with a live video feed. It will draw on detected markers, outlining them and displaying there local axes (z is the normal). Also, markers are labelled with their id and z position relative to the camera. In the top left corner, text shows the distance between markers if there are two markers detected. In the console, the program prints out the id of each marker and its 3d position relative to the camera and the rotation relative to its local axes. Clicking `q` will close the camera feed window, and the program will start presenting various graphs detail detailing data about the detections. Clicking `q` will close each graph, eventually terminating the program. The program will also save a file "results.txt" that for each detected marker will contain the average of the error (helpful for testing). 
- `aruco_dict` specifies the marker's dictionary [configuration details](#cv2arucogetpredefineddictionarycv2arucodict_6x6_250)
- `fileName` specifies the name of the file storing the calibration data for the camera in the current directory.
- `MARKER_SIZE` is the actual size in meters of the markers that are being detected. This needs to be accurate for the 3d positions to be accurate.
- `TRUE_DISTANCE` is the distance in meters between two fiducials. It is used for testing and is used in several of the graphs. For example, printing two markers out on paper with a known distance between them can make it easy to see any error in the detection.