import numpy as np

def calculate_square_area_3d(p1, p2, p3, p4):
    """
    Calculates the area of a square in 3D space given its four corner coordinates.

    Args:
        p1, p2, p3, p4: Tuples or lists of (x, y, z) coordinates for the corners,
                        provided in sequential order (e.g., clockwise or counter-clockwise).

    Returns:
        The surface area of the square.
    """
    # Convert points to numpy arrays
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    p4 = np.array(p4)

    # Form two adjacent vectors from a common vertex (e.g., p1)
    vector1 = p2 - p1
    vector2 = p4 - p1 # Assuming p1, p2, p3, p4 are in order around the perimeter

    # Verify if the points form a square by checking if the vectors are
    # 1. orthogonal (dot product is ~0)
    # 2. have the same magnitude (side lengths are equal)
    # This step is optional but good practice to ensure it is a square, not just a quadrilateral
    if not np.isclose(np.dot(vector1, vector2), 0) or not np.isclose(np.linalg.norm(vector1), np.linalg.norm(vector2)):
        print("Warning: The provided points do not form a valid square.")
        # For a general quadrilateral area, you would split it into two triangles and sum their areas.

    # The area of the parallelogram (which is a square here) formed by two adjacent vectors
    # is the magnitude of their cross product
    cross_prod = np.cross(vector1, vector2)
    area = np.linalg.norm(cross_prod)
    
    return area

# # Example usage:
# # A 2x2 square on the xy-plane at z=0
# corners_xy = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)]
# area_xy = calculate_square_area_3d(*corners_xy)
# print(f"Area of the square in the XY plane: {area_xy}") # Expected: 4.0

# # A 2x2 square tilted in 3D space
# corners_3d = [(0, 0, 0), (2, 0, 0), (2, 1, 2), (0, 1, 2)]
# area_3d = calculate_square_area_3d(*corners_3d)
# print(f"Area of the tilted square: {area_3d}") # Expected: 4.0 (length of (2,0,0) is 2, length of (0,1,2) is sqrt(5) - this is a rectangle, not a square. Corrected example below)

# # Corrected tilted square example (side length sqrt(2))
# corners_3d_square = [(0, 0, 0), (1, 1, 0), (0, 2, 0), (-1, 1, 0)]
# # This is not a square in 3D but on the xy plane. Let's make one in 3D.
# corners_3d_square_v2 = [(0, 0, 0), (1, 0, 1), (0, 1, 2), (-1, 1, 1)] # This is a diamond shape, not a square.

# # Let's use the example from a reliable source
# poly = [[0, 0, 0], [10, 0, 0], [10, 3, 4], [0, 3, 4]] # This is a rectangle with area 50
# area_rect = calculate_square_area_3d(*poly)
# print(f"Area of the rectangle: {area_rect}") # Expected: 50.0

