#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

class Light {
public:
    cv::RotatedRect rotatedRect;
    bool color;

    Light(const cv::RotatedRect& rect, bool col) : rotatedRect(rect), color(col) {}
};

class LightDetector {
public:
    LightDetector(const std::string& imagePath) : imagePath(imagePath) {}

    void run() {
        cv::Mat img_in = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (img_in.empty()) {
            std::cerr << "Error: Could not open or find the image." << std::endl;
            return;
        }

        cv::Mat img_binary = preprocessImage(img_in);
        std::vector<Light> lights = detectLights(img_binary);
        std::cout << "Detected " << lights.size() << " white rectangles." << std::endl;
        drawColorsText(img_in, lights);
        //drawLights(img_in, lights);
        drawPairedLights(img_in, pairLights(lights, 40, 200));
        display(img_in, "Detected Rotated Rectangles");
        cv::imwrite("images/detected.png", img_in);
    }

private:
    std::string imagePath;

    void display(const cv::Mat &img, const std::string &win_name = "Display") {
        cv::imshow(win_name, img);
        cv::waitKey(0);
        cv::destroyWindow(win_name);
    }

    cv::Mat preprocessImage(const cv::Mat &img) {
        cv::Mat img_gray, img_binary;
        cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
        cv::threshold(img_gray, img_binary, 220, 255, cv::THRESH_BINARY);
        //display(img_binary, "Binary Image");
        cv::imwrite("images/binary.png", img_binary);
        return img_binary;
    }

    bool getColor(const cv::Mat &img, const std::vector<cv::Point> &contour) {
        cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
        cv::drawContours(mask, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255), cv::FILLED);

        cv::Scalar meanColor = cv::mean(img, mask);
        if (meanColor[2] > meanColor[0]) {
            return 0; // Red
        } else if (meanColor[0] > meanColor[2]) {
            return 1; // Blue
        }
        return 0; // Default to Red if no dominant color
    }

    std::vector<Light> detectLights(const cv::Mat &img_binary) {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(img_binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<Light> lights;
        for (const auto& contour : contours) {
            cv::RotatedRect rotatedRect = cv::minAreaRect(contour);
            if (rotatedRect.size.area() < 150 ||
                rotatedRect.size.width / rotatedRect.size.height < 5 ||
                rotatedRect.size.width / rotatedRect.size.height > 20 ||
                abs(rotatedRect.angle) < 40) {
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
                // Check if the line is almost horizontal (within a small threshold)
                if (angleDiff < angleThreshold && distance < distanceThreshold && std::min(std::abs(centerLineAngle), std::abs(180 - std::abs(centerLineAngle))) < 10) {
                    pairedLights.push_back({lights[i], lights[j]});
                }
            }
        }
        return pairedLights;
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
    }
    void drawColorsText(cv::Mat &img, const std::vector<Light> &lights) {
        int count = 1;
        for (const auto& light : lights) {
            cv::putText(img, std::to_string(count) + ": " + (light.color ? "Blue" : "Red"), light.rotatedRect.center, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
            //output the data of the lights
            std::cout << count << ". Center: " << light.rotatedRect.center << " Color: " << (light.color ? "Blue" : "Red") << std::endl;
            count++;
        }
    }

    void drawLights(cv::Mat &img, const std::vector<Light> &lights) {
        for (const auto& light : lights) {
            cv::Point2f vertices[4];
            light.rotatedRect.points(vertices);
            for (int i = 0; i < 4; ++i) {
                cv::line(img, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
            }
        }
    }
};

int main() {
    LightDetector detector("images/image3.png");
    detector.run();
    return 0;
}
