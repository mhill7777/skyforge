import cv2
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
# Create board: 7 squares wide, 5 high. 
# Square length 0.04m (40mm), Marker length 0.02m (20mm)
board = cv2.aruco.CharucoBoard((9, 6), 24/900, 0.02, dictionary)

# Generate image to print (e.g., 1000x700 pixels)
pixelsLong=1000
pixelsTall=700
img = board.generateImage((pixelsLong, pixelsTall))
cv2.imwrite("my_charuco_board.png", img)
