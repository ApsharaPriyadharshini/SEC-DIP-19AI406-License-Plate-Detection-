# SEC-DIP-19AI406-License-Plate-Detection-
### Name: Apshara Priyadharshini M
### Register Number: 212225040026
## Aim
To implement a License Plate Detection system using OpenCV and Haar Cascade Classifier, draw bounding boxes, crop the detected region, and blur the license plate to improve privacy. The detection accuracy is improved by tuning Haar Cascade parameters.

## Software Used
Python 3.7 or above  
OpenCV (opencv-python)  
NumPy  
Matplotlib  
Jupyter Notebook (Anaconda)  
Haar Cascade File: haarcascade_russian_plate_number.xml

## Algorithm
Import necessary libraries such as OpenCV and Matplotlib  
Read the input vehicle image  
Convert the original image to grayscale for faster computation  
Load the Haar Cascade classifier for license plate detection  
Detect license plate using detectMultiScale function  
Draw rectangle around detected area  
Crop the detected region using numpy slicing with (x, y, w, h) values  
Apply median blurring on the cropped region  
Replace the original region with blurred version  
Display final result using Matplotlib

## Program
```python
import cv2
import matplotlib.pyplot as plt

def display(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,6))
    plt.imshow(img_rgb)
    plt.axis('off')

img = cv2.imread("DATA/car_plate.jpg")
plate_cascade = cv2.CascadeClassifier("DATA/haarcascades/haarcascade_russian_plate_number.xml")

def detect_and_blur_plate(img):
    img_copy = img.copy()
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in plates:
        roi = img_copy[y:y+h, x:x+w]
        blurred_roi = cv2.medianBlur(roi, 15)
        img_copy[y:y+h, x:x+w] = blurred_roi

    return img_copy

result = detect_and_blur_plate(img)
display(result)
```

## Output
<img width="825" height="443" alt="image" src="https://github.com/user-attachments/assets/af3143d8-2164-420d-b55e-59364d842db8" />
<img width="816" height="371" alt="image" src="https://github.com/user-attachments/assets/31442cad-4f2a-4a3a-b509-c0b59de9d895" />
<img width="878" height="411" alt="image" src="https://github.com/user-attachments/assets/b5be05e6-58c5-4696-9a78-824e2d0c3d69" />



## Modification Done

Parameter tuning was performed by adjusting scaleFactor and minNeighbors values in detectMultiScale to improve accuracy and reduce false detections. Median blur was applied to protect license plate information.

## Result

The License Plate Detection system was successfully implemented using OpenCV and Haar Cascade. The detected license plate region was blurred using median filtering. The modified values improved overall detection performance and output quality.

## Conclusion

This workshop demonstrates how classical computer vision methods like Haar Cascades can be used for real-time applications such as automated toll systems, smart parking, and traffic surveillance. Proper preprocessing and parameter tuning significantly improve detection results.
