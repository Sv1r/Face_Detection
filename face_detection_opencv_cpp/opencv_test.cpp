#include <iostream>
#include <string> 
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main() {
    VideoCapture cam(0);
    if (!cam.isOpened()) {
        throw runtime_error("Error");
    }

    namedWindow("Window");
    while (true) {
        // Get image from cam
        Mat image;
        cam >> image;
        resize(image, image, Size(), 1, 1, INTER_LINEAR);
        flip(image, image, 1);
        // Convert image to gray
        Mat gray;
        cvtColor(image, gray, COLOR_BGR2GRAY);
        // Load Face Detector
        CascadeClassifier face_cascade("./haarcascade_frontalface_default.xml");
        // Detect faces
        vector<Rect> faces;
        face_cascade.detectMultiScale(gray, faces);
        for (size_t i = 0; i < faces.size(); i++)
        {
            rectangle(image, faces[i], Scalar(0, 0, 255), 2);
            // Add image coordinates on rectangles
            string x_center = to_string(faces[i].x + faces[i].width / 2);
            string y_center = to_string(faces[i].y + faces[i].height / 2);
            // Put text on image
            putText(
                image,
                (x_center + " " + y_center),
                Point(faces[i].x, faces[i].y - 5),
                FONT_HERSHEY_DUPLEX,
                0.6,
                Scalar(0, 0, 255)
            );
        }

        imshow("Image", image);
        if (waitKey(1) == 27) break;
    }
    return 0;
}