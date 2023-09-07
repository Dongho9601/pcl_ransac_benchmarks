#include "fitting3D.hpp"

template <typename modelType>
void Fitter3D::runFitting(PointCloudPtr& cloudCopy, modelType& model) {
    // pcl::RandomSampleConsensus<PointCloudType> ransac(model);
    // ransac.setMaxIterations(m_maxIterations);
    // ransac.setProbability(m_threshold);
    // ransac.setDistanceThreshold(m_delta);
    // ransac.computeModel();

    // Eigen::VectorXf modelCoefficients;
    // ransac.getModelCoefficients(modelCoefficients);
    // getBestModelCoefficients(modelCoefficients);

    // // remove inliners
    // if (m_remainingPointsRatio == 1) return;
    // pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    // ransac.getInliers(inliers->indices);
    // pcl::ExtractIndices<PointCloudType> extract;
    // extract.setInputCloud(cloudCopy);
    // extract.setIndices(inliers);
    // extract.setNegative(true);
    // extract.filter(*cloudCopy);
}

void Fitter3D::run(const PointCloudPtr& cloud) {
    // copy the input cloud
    PointCloudPtr cloudCopy(new PointCloud);
    if (m_remainingPointsRatio != 1)
        pcl::copyPointCloud(*cloud, *cloudCopy);
    else
        cloudCopy = cloud;

    while (cloudCopy->size() >= cloud->size() * m_remainingPointsRatio) {
        if (m_application == "line" && m_device == "CPU") {
            // pcl::SampleConsensusModelLine<PointCloudType>::Ptr 
            //     model(new pcl::SampleConsensusModelLine<PointCloudType>(cloudCopy));
            // runFitting<pcl::SampleConsensusModelLine<PointCloudType>::Ptr>(cloudCopy, model);
            ;

        } else if (m_application == "plane" && m_device == "CPU") {
            ;

        } else if (m_application == "circle" && m_device == "CPU") {
            ;

        } else if (m_application == "sphere" && m_device == "CPU") {
            ;

        } else if (m_application == "cylinder" && m_device == "CPU") {
            ;

        } else if (m_device == "GPU") {
            runFittingWithCUDA(cloudCopy);

        } else {
            std::cerr << "Invalid application" << std::endl;
            abort();
        }

        if (m_remainingPointsRatio == 1) break;
    }

}

cv::Mat Fitter3D::draw3DImage(const PointCloudPtr& cloud,
                              const float step,
                              const int defaultWidth, 
                              const int defaultHeight)
{
    cv::Scalar bgColor = cv::Scalar(255, 255, 255);
    if (cloud->size() < 2) {
        return cv::Mat(defaultHeight, defaultWidth, CV_8UC3, bgColor);
    }

    // project the point cloud to the plane, x+y+z=2
    // and find the min and max points
    std::vector<cv::Point2f> points;
    float min_u = 1000.0f, min_v = 1000.0f;
    float max_u = -1000.0f, max_v = -1000.0f;
    for (const auto& point : cloud->points) {
        float u = point.y - point.x / 1.414f;
        float v = point.z - point.x / 1.414f;
        if (u < min_u) min_u = u;
        if (u > max_u) max_u = u;
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        points.push_back(cv::Point2f(u, v));
    }
    
    int width = (max_u - min_u) / step;
    int height = (max_v - min_v) / step;

    cv::Mat image(height, width, CV_8UC3, bgColor);
    
    // draw the x, y, and z axis
    cv::line(image, cv::Point(0, height - 1), cv::Point((width - 1)/3, 2*(height-1)/3), cv::Scalar(64, 64, 64), 1, cv::LINE_AA);
    cv::line(image, cv::Point((width - 1)/3, 0), cv::Point((width - 1)/3, 2*(height-1)/3), cv::Scalar(64, 64, 64), 1, cv::LINE_AA);
    cv::line(image, cv::Point(width - 1, 2*(height - 1)/3), cv::Point((width - 1)/3, 2*(height-1)/3), cv::Scalar(64, 64, 64), 1, cv::LINE_AA);

    for (const auto& point : points) {
        int x = (point.x - min_u) / step;
        int y = (point.y - min_v) / step;
        cv::circle(image, cv::Point(x, y), 1, cv::Scalar(0, 0, 0), -1, cv::LINE_AA);
    }

    return image;
}