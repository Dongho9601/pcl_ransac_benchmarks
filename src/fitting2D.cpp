#include "fitting2D.hpp"

// do RANSAC algorithm until the remaining points are 
// less than the number of points times remainingPointsRatio
template <typename modelType>
void Fitter2D::runFitting(PointCloudPtr& cloudCopy, modelType& model) {
    pcl::RandomSampleConsensus<PointCloudType> ransac(model);
    ransac.setMaxIterations(m_maxIterations);
    ransac.setProbability(m_threshold);
    ransac.setDistanceThreshold(m_delta);
    ransac.computeModel();

    Eigen::VectorXf modelCoefficients;
    ransac.getModelCoefficients(modelCoefficients);
    getBestModelCoefficients(modelCoefficients);

    // remove inliners
    // if (remainingPointsRatio == 1) return;
    // pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    // ransac.getInliers(inliers->indices);
    // pcl::ExtractIndices<PointCloudType> extract;
    // extract.setInputCloud(cloudCopy);
    // extract.setIndices(inliers);
    // extract.setNegative(true);
    // extract.filter(*cloudCopy);
}

void Fitter2D::run(const PointCloudPtr& cloud) {
    // copy the input cloud
    PointCloudPtr cloudCopy(new PointCloud);
    if (m_remainingPointsRatio != 1)
        pcl::copyPointCloud(*cloud, *cloudCopy);
    else
        cloudCopy = cloud;

    while (cloudCopy->size() >= cloud->size() * m_remainingPointsRatio) {
        if (m_application == "line" && m_device == "CPU") {
            pcl::SampleConsensusModelLine<PointCloudType>::Ptr 
                model(new pcl::SampleConsensusModelLine<PointCloudType>(cloudCopy));
            runFitting<pcl::SampleConsensusModelLine<PointCloudType>::Ptr>(cloudCopy, model);

        } else if (m_application == "circle" && m_device == "CPU") {
            pcl::SampleConsensusModelCircle2D<PointCloudType>::Ptr 
                model(new pcl::SampleConsensusModelCircle2D<PointCloudType>(cloudCopy));
            runFitting<pcl::SampleConsensusModelCircle2D<PointCloudType>::Ptr>(cloudCopy, model);
        
        } else if (m_device == "GPU") {
            runFittingWithCUDA(cloudCopy);

        } else {
            std::cerr << "Invalid application" << std::endl;
            abort();
        }

        if (m_remainingPointsRatio == 1) break;
    }

}

cv::Mat Fitter2D::draw2DImage(const PointCloudPtr& inCloud,
                              const float step,
                              const int defaultWidth, 
                              const int defaultHeight)
{
    cv::Scalar bgColor = cv::Scalar(167, 167, 167);
    if (inCloud->size() < 2) {
        return cv::Mat(defaultHeight, defaultWidth, CV_8UC3, bgColor);
    }

    pcl::PointXYZ minP, maxP;
    pcl::getMinMax3D<pcl::PointXYZ>(*inCloud, minP, maxP);

    int height = (maxP.y - minP.y) / step;
    int width = (maxP.x - minP.x) / step;

    cv::Mat image(height, width, CV_8UC3, bgColor);

    for (const auto& point : inCloud->points) {
        int x = (point.x - minP.x) / step;
        int y = (point.y - minP.y) / step;
        image.ptr<cv::Vec3b>(y)[x] = cv::Vec3b(0, 0, 255);
        cv::circle(image, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
    }

    if (m_application == "line") {
        // inliner(x,y) direction(x,y)
        for (const auto& model : m_bestModelCoefficients) {
            float minT = std::abs((minP.x-model[0])/model[3]) > abs((minP.y-model[1])/model[4]) ?
                        (minP.x-model[0])/model[3] : (minP.y-model[1])/model[4];
            float maxT = std::abs((maxP.x-model[0])/model[3]) > abs((maxP.y-model[1])/model[4]) ?
                        (maxP.x-model[0])/model[3] : (maxP.y-model[1])/model[4] ;
            float x1 = (model[0]+minT*model[3]-minP.x)/step;
            float y1 = (model[1]+minT*model[4]-minP.y)/step;
            float x2 = (model[0]+maxT*model[3]-minP.x)/step;
            float y2 = (model[1]+maxT*model[4]-minP.y)/step;
            cv::line(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        }

    } else if (m_application == "circle") {
        // origin(x,y) R
        for (const auto& model : m_bestModelCoefficients) {
            float x1 = (model[0]-minP.x)/step;
            float y1 = (model[1]-minP.y)/step;
            float r = model[2]/step;
            cv::circle(image, cv::Point(x1, y1), r, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        }

    } else {
        std::cerr << "Invalid application" << std::endl;
        abort();
    }

    return image;
}