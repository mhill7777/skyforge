import cv2
import os # Import the operating system module

# Define the desired save directory
save_directory = "../CalibrationImages" # <-- CHANGE THIS to your desired path
imageNum=0 # starting value
 

# Ensure the directory exists before trying to save
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
    print(f"Created directory: {save_directory}")



# Initialize the camera
cam = cv2.VideoCapture(0)
cv2.namedWindow("Camera Feed - Press SPACE to capture, ESC to exit")

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("Camera Feed - Press SPACE to capture, ESC to exit", frame)

    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed, exit loop
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed, save image to the specified full path

        # Combine the directory path and the file name
        # This handles the correct use of forward/backward slashes for Windows/macOS/Linux
        full_save_path = os.path.join(save_directory, "image"+str(imageNum)+".png")
        imageNum=imageNum+1
        cv2.imwrite(full_save_path, frame)
        print(f"Image saved to: {full_save_path}") # Print the full path
        #break # Exit loop after capturing

# Release the camera and destroy all windows
cam.release()
cv2.destroyAllWindows()
