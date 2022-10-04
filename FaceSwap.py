#The code below is not fully functional. There is still some errors to correct.
#Meanwhile, the reader can see the progress made so far.






import cv2 as cv
import dlib
import numpy as np
import time
img1 = cv.imread(input("Type in your exact souce image name with extension" + "\n"))
img2 = cv.imread(input("Type in your exact destination image name with extension" + "\n"))
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
mask = np.zeros_like(img1_gray)
land_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
img1_face = land_detector(img1_gray)
for face in img1_face:
    landmarks = predictor(img1_gray, face)
    points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        points.append((x, y))
        face_point = np.array(points, np.int32)
        convexhull = cv.convexHull(face_point)
img2_face = land_detector(img2_gray)
for face in img2_face:
    landmarks = predictor(img2_gray, face)
    points2 = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        points2.append((x, y))
rectangle = cv.boundingRect(convexhull)
divide_2d = cv.Subdiv2D(rectangle)
divide_2d.insert(landmarks_points)
list_triangles_split = divide_2d.getTriangleList()
list_triangles_split = np.array(split_triangle, dtype=np.int32)
cv.fillConvexPoly(mask, convexhull, 255)
face_image_1 = cv.bitwise_and(img1, img1, mask=mask)
face_points2 = np.array(points2, np.int32)
convexhull2 = cv.convexHull(face_points2)
def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index
index_collection = []
for edge in list_triangles_split:
    first = (edge[0], edge[1])
    second = (edge[2], edge[3])
    third = (edge[4], edge[5])
index_edge1 = np.where((points == first).all(axis=1))
index_edge1 = extract_index_nparray(index_edge1)
index_edge2 = np.where((points == pt2).all(axis=1))
index_edge2 = extract_index_nparray(index_edge2)
index_edge3 = np.where((points == pt3).all(axis=1))
index_edge3 = extract_index_nparray(index_edge3)
if index_edge1 is not None and index_edge2 is not None and index_edge3 is not None:
    triangle = [index_edge1, index_edge2, index_edge3]
    index_collection.append(triangle)
source_mask = np.zeros_like(img1_gray)
new_face = np.zeros_like(img2)
for index in index_collection:
    tri_one = points[index[0]]
    tri_two = points[index[1]]
    tri_three = points[index[2]]
    triangle1 = np.array([tri_one, tri_two, tri_three], np.int32)
    first_rect = cv.boundingRect(triangle1)
    (x, y, w, h) = first_rect
    cropped_triangle = img1[y: y + h, x: x + w]
    cropped_tr1_mask = np.zeros((h, w), np.uint8)
    pts = np.array([[tri_one[0] - x, tri_one[1] - y],
                       [tri_two[0] - x, tri_two[1] - y],
                       [tri_three[0] - x, tri_three[1] - y]], np.int32)
    cv.fillConvexPoly(cropped_tr1_mask, pts, 255)
    cv.line(source_mask, tri_one, tri_two, 255)
    cv.line(source_mask, tri_two, tri_three, 255)
    cv.line(source_mask, tri_one, tri_three, 255)
tri2_one = points2[index[0]]
tri2_two = points2[index[1]]
tri2_three = points2[index[2]]
triangle2 = np.array([tri2_one, tri2_two, tri2_three], np.int32)
second_rect = cv.boundingRect(triangle2)
(x, y, w, h) = second_rect
cropped = np.zeros((h, w), np.uint8)
points2 = np.array([[tri2_one[0] - x, tri2_one[1] - y],
                    [tri2_two[0] - x, tri2_two[1] - y],
                    [tri2_three[0] - x, tri2_three[1] - y]], np.int32)
cv.fillConvexPoly(cropped, points2, 255)
points = np.float32(points)
points2 = np.float32(points2)
transform = cv.getAffineTransform(points, points2)
warping = cv.warpAffine(cropped_triangle, transform, (w, h))
warping = cv.bitwise_and(warping, warping, mask=cropped)
ht, wt, filters = img2.shape
img2_face = np.zeros((ht, wt, filters), np.uint8)
facial_area = img2_face[y: y + h, x: x + w]
facial_area_gray = cv.cvtColor(facial_area, cv.COLOR_BGR2GRAY)
triangle_mask = cv.threshold(facial_area_gray, 1, 255, cv.THRESH_BINARY_INV)
warping = cv.bitwise_and(warping, warping, mask=triangle_mask)
facial_area = cv.add(facial_area, warping)
img2_face[y: y + h, x: x + w] = facial_area
final_mask = np.zeros_like(img2_gray)
head_mask = cv.fillConvexPoly(final_mask, convexhull2, 255)
final_mask = cv.bitwise_not(head_mask)
combine = cv.bitwise_and(img2, img2, mask=final_mask)
output = cv.add(combine, img2_face)
(x, y, w, h) = cv.boundingRect(convexhull2)
seamless= (int((x + x + w) / 2), int((y + y + h) / 2))
seamlessclone = cv.seamlessClone(output, img2, head_mask, seamless, cv.NORMAL_CLONE)
cv.imshow("seamlessclone", seamlessclone)
cv.waitKey(0)
cv.destroyAllWindows()
