import cv2
import numpy as np
import operator
from imutils import contours
import imutils
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from models.modelM5 import ModelM5
import torch

# enable GPU usage ------------------------------------------------------------#
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda == False:
    print("WARNING: CPU will be used.")
    exit(0)

#Load the model
new_model = tf.keras.models.load_model('Digit_Recognizer.h5')
#new_model = ModelM5().to(device)

#Parameters for Warping the image
margin = 10
case = 28 + 2*margin
perspective_size = 9*case

useStaticImageOnly = True
cap = cv2.VideoCapture(0)
if useStaticImageOnly:
    cap = cv2.imread('../data/sudoku.png', cv2.IMREAD_UNCHANGED)

flag = 0
ans = 0


def findNextCellToFill(grid, i, j):
    for x in range(i, 9):
        for y in range(j, 9):
            if grid[x][y] == 0:
                return x, y
    for x in range(0, 9):
        for y in range(0, 9):
            if grid[x][y] == 0:
                return x, y
    return -1, -1


def isValid(grid, i, j, e):
    try:
        rowOk = all([e != grid[i][x] for x in range(9)])

        #If row is valid, try to check the columns.
        if rowOk:
            columnOk = all([e != grid[x][j] for x in range(9)])
            if columnOk:
                # finding the top left x,y co-ordinates of the section containing the i,j cell
                secTopX, secTopY = 3 * (i // 3), 3 * (j // 3)  # floored quotient should be used here.
                for x in range(secTopX, secTopX + 3):
                    for y in range(secTopY, secTopY + 3):
                        if grid[x][y] == e:
                            return False
                return True
    except IndexError:
        pass
    return False


def solveSudoku(grid, i=0, j=0):
    i, j = findNextCellToFill(grid, i, j)
    if i == -1:
        return True
    for e in range(1, 10):
        if isValid(grid, i, j, e):
            grid[i][j] = e
            if solveSudoku(grid, i, j):
                return True
            # Undo the current cell for backtracking
            grid[i][j] = 0
    return False


'''Capture the contour from the capture'''
#Gets predicted digit nested list
def display_predList(predList):
    predicted_digits = []
    for i in range(len(predList)):
        for j in range(len(predList)):
            predicted_digits.append(predList[j][i])
    return predicted_digits


def processFrameForContour(frame):
    #Process the frame to find contour
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)

    #Get all of the contrours in the frame.
    contours_, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours_


def FindLargestContour(contours):
    # Find the largest contour(Sudoku Grid)
    maxArea = 0
    contour = None
    for c in contours:
        area = cv2.contourArea(c)

        if area > 25000:
            peri = cv2.arcLength(c, True)
            polygon = cv2.approxPolyDP(c, 0.01 * peri, True)

            if area > maxArea and len(polygon) == 4:
                contour = polygon
                maxArea = area
    return contour, maxArea


# Draw the contour and extract Sudoku Grid
def ExtractGrid(contour, frame):
    if contour is None:
        return

    global perspective_size

    cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)
    points = np.vstack(contour).squeeze()
    points = sorted(points, key=operator.itemgetter(1))

    if points[0][0] < points[1][0]:
        if points[3][0] < points[2][0]:
            pts1 = np.float32([points[0], points[1], points[3], points[2]])
        else:
            pts1 = np.float32([points[0], points[1], points[2], points[3]])
    else:
        if points[3][0] < points[2][0]:
            pts1 = np.float32([points[1], points[0], points[3], points[2]])
        else:
            pts1 = np.float32([points[1], points[0], points[2], points[3]])

    pts2 = np.float32([[0, 0], [perspective_size, 0], [0, perspective_size], [perspective_size, perspective_size]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    perspective_window = cv2.warpPerspective(p_frame, matrix, (perspective_size, perspective_size))
    result = perspective_window.copy()

    # Process the extracted Sudoku Grid
    p_window = cv2.cvtColor(perspective_window, cv2.COLOR_BGR2GRAY)
    p_window = cv2.GaussianBlur(p_window, (5, 5), 0)
    p_window = cv2.adaptiveThreshold(p_window, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    p_window = cv2.morphologyEx(p_window, cv2.MORPH_CLOSE, vertical_kernel)
    lines = cv2.HoughLinesP(p_window, 1, np.pi / 180, 120, minLineLength=40, maxLineGap=10)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(perspective_window, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Invert the grid for digit recognition
    invert = 255 - p_window
    invert_window = invert.copy()

    invert_window = invert_window / 255

    return result, invert, invert_window, matrix, p_window

while True:
    frame = None
    if useStaticImageOnly:
        frame = cap
    else:
        unusedRet, frame = cap.read()
    p_frame = frame.copy()

    contours = processFrameForContour(frame)
    contour, maxArea = FindLargestContour(contours)
    if contour is not None:
        i = 0
        result, invert, invert_window, matrix, p_window = ExtractGrid(contour, frame)

        # Check if the answer has been already predicted or not
        # If not predict the answer
        # Else only get the cell regions
        cell = 1
        if flag != 1:
            predicted_digits = []
            pixels_sum = []

        # To get individual cells
        for y in range(9):
            predicted_line = []
            for x in range(9):
                y2min = y * case + margin
                y2max = (y + 1) * case - margin
                x2min = x * case + margin
                x2max = (x + 1) * case - margin

                # Obtained Cell
                image = invert_window[y2min:y2max, x2min:x2max]
                if cell:
                    cell = 0
                    #cv2.imshow("HELP ME GOD EMPEROR!", image)

                # Process the cell to feed it into model
                img = cv2.resize(image, (28, 28))
                img = img.reshape((1, 28, 28, 1))

                # Get sum of all the pixels in the cell
                # If sum value is large it means the cell is blank
                pixel_sum = np.sum(img)
                pixels_sum.append(pixel_sum)

                # Predict the digit in the cell
                pred = new_model.predict(img)
                predicted_digit = pred.argmax()

                # For blank cells set predicted digit to 0
                if pixel_sum > 775.0:
                    predicted_digit = 0

                predicted_line.append(predicted_digit)

                # If we already have predicted result, display it on window
                if flag == 1:
                    ans = 1
                    x_pos = int((x2min + x2max) / 2) + 10
                    y_pos = int((y2min + y2max) / 2) - 5
                    image = cv2.putText(result, str(pred_digits[i]), (y_pos, x_pos), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (255, 0, 0), 2, cv2.LINE_AA)

                i = i + 1

            # Get predicted digit list
            if flag != 1:
                predicted_digits.append(predicted_line)
            print(str(predicted_digits))

            # Get solved Sudoku
            ans = solveSudoku(predicted_digits)
            if ans == True:
                flag = 1
                pred_digits = display_predList(predicted_digits)

                # Display the final result
                if ans == 1:
                    cv2.imshow("Result", result)
                    frame = cv2.warpPerspective(result, matrix, (perspective_size, perspective_size),
                                                flags=cv2.WARP_INVERSE_MAP)

            cv2.imshow("frame", frame)
            cv2.imshow('P-Window', p_window)
            cv2.imshow('Invert', invert)

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

