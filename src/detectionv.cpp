#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <iostream>

class Light {
public:
    cv::RotatedRect rotatedRect;
    bool color;

    Light(const cv::RotatedRect& rect, bool col) : rotatedRect(rect), color(col) {}
};

class LightDetector {
public:
    LightDetector(const std::string& videoPath) : videoPath(videoPath) {}

    void run() {
        cv::VideoCapture cap(videoPath);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open video file." << std::endl;
            return;
        }

        cv::Mat frame;
        cv::VideoWriter outputVideo;
        int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        outputVideo.open("output.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(frame_width, frame_height), true);

        if (!outputVideo.isOpened()) {
            std::cerr << "Error: Could not open the output video for write." << std::endl;
            return;
        }

        while (cap.read(frame)) {
            cv::Mat img_binary = preprocessImage(frame);
            std::vector<Light> lights = detectLights(img_binary);
            std::vector<std::pair<Light, Light>> pairedLights = pairLights(lights, 40, 200);
            drawPairedLights(frame, pairedLights);
            cv::imshow("Detected Rotated Rectangles", frame);
            outputVideo.write(frame);

            if (cv::waitKey(30) >= 0) break; // Press any key to exit
        }

        cap.release();
        outputVideo.release();
        cv::destroyAllWindows();
    }

    cv::Mat preprocessImage(const cv::Mat &img) {
        cv::Mat img_gray, img_binary;
        cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
        cv::threshold(img_gray, img_binary, 220, 255, cv::THRESH_BINARY);
        return img_binary;
    }

    std::vector<Light> detectLights(const cv::Mat &img_binary) {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(img_binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<Light> lights;
        for (const auto& contour : contours) {
            cv::RotatedRect rotatedRect = cv::minAreaRect(contour);
            if (rotatedRect.size.area() < 100 || // Lowered area threshold
                rotatedRect.size.width / rotatedRect.size.height < 3 || // Lowered aspect ratio threshold
                rotatedRect.size.width / rotatedRect.size.height > 25 || // Increased aspect ratio threshold
                abs(rotatedRect.angle) < 30) { // Lowered angle threshold
                continue;
            }
            lights.push_back(Light(rotatedRect, getColor(img_binary, contour)));
        }
        return lights;
    }

    std::vector<std::pair<Light, Light>> pairLights(const std::vector<Light>& lights, double angleThreshold, double distanceThreshold) {
        std::vector<std::pair<Light, Light>> pairedLights;
        for (size_t i = 0; i < lights.size(); ++i) {
            for (size_t j = i + 1; j < lights.size(); ++j) {
                double angleDiff = std::abs(lights[i].rotatedRect.angle - lights[j].rotatedRect.angle);
                double distance = cv::norm(lights[i].rotatedRect.center - lights[j].rotatedRect.center);

                // Calculate the angle of the line connecting the centers of the two lights
                double deltaX = lights[j].rotatedRect.center.x - lights[i].rotatedRect.center.x;
                double deltaY = lights[j].rotatedRect.center.y - lights[i].rotatedRect.center.y;
                double centerLineAngle = std::atan2(deltaY, deltaX) * 180.0 / CV_PI;
                std::cout << "Angle Difference: " << angleDiff 
                          << ", Distance: " << distance 
                          << ", Center Line Angle: " << centerLineAngle 
                          << std::endl;
                // Check if the line is almost horizontal (within a larger threshold)
                if (angleDiff < angleThreshold && distance < distanceThreshold && std::min(std::abs(centerLineAngle), std::abs(180 - std::abs(centerLineAngle))) < 20) {
                    pairedLights.push_back({lights[i], lights[j]});
                }
            }
        }
        return pairedLights;
    }

    void drawPairedLightsLine(cv::Mat &img, const std::vector<std::pair<Light, Light>> &pairedLights) {
        for (const auto& pair : pairedLights) {
            cv::line(img, pair.first.rotatedRect.center, pair.second.rotatedRect.center, cv::Scalar(0, 255, 0), 2);
        }
    }
    void drawPairedLights(cv::Mat &img, const std::vector<std::pair<Light, Light>> &pairedContours) {
        std::cout << "Detected " << pairedContours.size() << " pairs of white rectangles." << std::endl;    
        for (size_t i = 0; i < pairedContours.size(); ++i) {
            const auto& pair = pairedContours[i];
            cv::Scalar color(128 + (i * 50) % 128, 255 - (i * 50) % 128, 128 + (i * 50) % 128);

            cv::Point2f vertices1[4];
            pair.first.rotatedRect.points(vertices1);
            for (int j = 0; j < 4; ++j) {
                cv::line(img, vertices1[j], vertices1[(j + 1) % 4], color, 2);
            }

            cv::Point2f vertices2[4];
            pair.second.rotatedRect.points(vertices2);
            for (int j = 0; j < 4; ++j) {
                cv::line(img, vertices2[j], vertices2[(j + 1) % 4], color, 2);
            }
        }
        drawPairedLightsLine(img, pairedContours);
    }
    bool getColor(const cv::Mat &img_binary, const std::vector<cv::Point> &contour) {
        cv::Mat mask = cv::Mat::zeros(img_binary.size(), CV_8UC1);
        cv::drawContours(mask, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255), cv::FILLED);

        cv::Scalar meanColor = cv::mean(img_binary, mask);
        if (meanColor[2] > meanColor[0]) {
            return 0; // Red
        } else if (meanColor[0] > meanColor[2]) {
            return 1; // Blue
        }
        return 0; // Default to Red if no dominant color
    }

private:
    std::string videoPath;
};

int main() {
    std::string videoPath = "videos/normal.avi";
    LightDetector detector(videoPath);
    detector.run();
    return 0;
}