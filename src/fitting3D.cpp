#include "fitting3D.hpp"

template <typename modelType>
void Fitter3D::runFitting(PointCloudPtr& cloudCopy, modelType& model) {
    pcl::RandomSampleConsensus<PointCloudType> ransac(model);
    ransac.setMaxIterations(m_maxIterations);
    ransac.setProbability(m_threshold);
    ransac.setDistanceThreshold(m_delta);
    ransac.computeModel();

    Eigen::VectorXf modelCoefficients;
    ransac.getModelCoefficients(modelCoefficients);
    getBestModelCoefficients(modelCoefficients);

    // remove inliners
    if (m_remainingPointsRatio == 1) return;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    ransac.getInliers(inliers->indices);
    pcl::ExtractIndices<PointCloudType> extract;
    extract.setInputCloud(cloudCopy);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*cloudCopy);
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
            pcl::SampleConsensusModelLine<PointCloudType>::Ptr 
                model(new pcl::SampleConsensusModelLine<PointCloudType>(cloudCopy));
            runFitting<pcl::SampleConsensusModelLine<PointCloudType>::Ptr>(cloudCopy, model);

        } else if (m_application == "plane" && m_device == "CPU") {
            pcl::SampleConsensusModelPlane<PointCloudType>::Ptr 
                model(new pcl::SampleConsensusModelPlane<PointCloudType>(cloudCopy));
            runFitting<pcl::SampleConsensusModelPlane<PointCloudType>::Ptr>(cloudCopy, model);

        } else if (m_application == "circle" && m_device == "CPU") {
            pcl::SampleConsensusModelCircle3D<PointCloudType>::Ptr 
                model(new pcl::SampleConsensusModelCircle3D<PointCloudType>(cloudCopy));
            runFitting<pcl::SampleConsensusModelCircle3D<PointCloudType>::Ptr>(cloudCopy, model);

        } else if (m_application == "sphere" && m_device == "CPU") {
            pcl::SampleConsensusModelSphere<PointCloudType>::Ptr 
                model(new pcl::SampleConsensusModelSphere<PointCloudType>(cloudCopy));
            runFitting<pcl::SampleConsensusModelSphere<PointCloudType>::Ptr>(cloudCopy, model);

        } else if (m_application == "cylinder" && m_device == "CPU") {
            pcl::SampleConsensusModelCylinder<PointCloudType, pcl::Normal>::Ptr 
                model(new pcl::SampleConsensusModelCylinder<PointCloudType, pcl::Normal>(cloudCopy));
            runFitting<pcl::SampleConsensusModelCylinder<PointCloudType, pcl::Normal>::Ptr>(cloudCopy, model);

        } else if (m_device == "GPU") {
            runFittingWithCUDA(cloudCopy);

        } else {
            std::cerr << "Invalid application" << std::endl;
            abort();
        }

        if (m_remainingPointsRatio == 1) break;
    }

}

float projection(float YorZ, int coeff, float X) {return YorZ / X * coeff;}

cv::Mat Fitter3D::draw3DImage(const PointCloudPtr& cloud,
                              const float step,
                              const int defaultWidth, 
                              const int defaultHeight)
{
    cv::Scalar bgColor = cv::Scalar(255, 255, 255, 255);
    if (cloud->size() < 2)
        return cv::Mat(defaultHeight, defaultWidth, CV_8UC4, bgColor);

    // project the point cloud to the plane, x+y+z=2
    // and find the min and max points
    std::vector<cv::Point2f> projectedPoints;
    float min_u = 1000.0f, min_v = 1000.0f;
    float max_u = -1000.0f, max_v = -1000.0f;
    for (const auto& point : cloud->points) {
        float u = point.x / point.z;
        float v = point.y / point.z;
        if (u < min_u) min_u = u;
        if (u > max_u) max_u = u;
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        projectedPoints.push_back(cv::Point2f(u, v));
    }
    std::cout << "min_u: " << min_u/step << " max_u: " << max_u/step << std::endl;
    std::cout << "min_v: " << min_v/step << " max_v: " << max_v/step << std::endl;
    
    int width = (max_u - min_u) / step;
    int height = (max_v - min_v) / step;
    int diagonal = std::sqrt(width * width + height * height);

    cv::Mat image(height, width, CV_8UC4, bgColor);
    std::cout << "width: " << width << " height: " << height << std::endl;
    
    if (m_application == "line") {
        for (const auto& model : m_bestModelCoefficients) {
            float minT = (model[0]-model[2]*min_u) / (model[5]*min_u - model[3]);
            float maxT = (model[0]-model[2]*max_u) / (model[5]*max_u - model[3]);
            // min point
            float x1 = (model[0] + minT * model[3]);
            float y1 = (model[1] + minT * model[4]);
            float z1 = (model[2] + minT * model[5]);
            x1 = (x1/z1 - min_u) / step;
            y1 = (y1/z1 - min_v) / step;
            // max point
            float x2 = (model[0]+ maxT * model[3]);
            float y2 = (model[1]+ maxT * model[4]);
            float z2 = (model[2]+ maxT * model[5]);
            x2 = (x2/z2 - min_u) / step;
            y2 = (y2/z2 - min_v) / step;
            // projection
            std::cout << "x1: " << x1 << " y1: " << y1 << std::endl;
            std::cout << "x2: " << x2 << " y2: " << y2 << std::endl;
            cv::line(image, cv::Point( x1,y1 ), cv::Point( x2,y2 ), cv::Scalar(0, 0, 255, 255), 1, cv::LINE_AA);
        }

    } else if (m_application == "plane") {
        for (const auto& model : m_bestModelCoefficients) {
            // top bottom left right
            cv::Point2f p1(0,1000.0f), p2(0,-1000.0f) , p3(1000.0f,0), p4(-1000.0f,0);
            for (const auto& point : cloud->points) {
                if (model[0]*point.x + model[1]*point.y + model[2]*point.z + model[3] > m_delta) continue;
                float u = point.x / point.z;
                float v = point.y / point.z;
                if (v < p1.y) p1 = cv::Point2f(u, v); 
                if (v > p2.y) p2 = cv::Point2f(u, v);
                if (u < p3.x) p3 = cv::Point2f(u, v);
                if (u > p4.x) p4 = cv::Point2f(u, v);
            }

            std::vector<cv::Point> four_points;
            four_points.push_back(cv::Point2f( (p1.x-min_u)/step, (p1.y-min_v)/step ));
            four_points.push_back(cv::Point2f( (p3.x-min_u)/step, (p3.y-min_v)/step ));
            four_points.push_back(cv::Point2f( (p2.x-min_u)/step, (p2.y-min_v)/step ));
            four_points.push_back(cv::Point2f( (p4.x-min_u)/step, (p4.y-min_v)/step ));
            
            // draw a filled polygon in the middle of the image with the color of read with 50% opacity
            cv::fillConvexPoly(image, four_points, cv::Scalar(0, 0, 255, 255), cv::LINE_AA);
        }

    } else if (m_application == "circle") {
        // for (const auto& model : m_bestModelCoefficients) {
        //     // origin(x,y,z) R normal(x,y,z)
        //     float u = projection(model[1], model[0]);
        //     float v = projection(model[2], model[0]);
        //     float x1 = (u-min_u)/step;
        //     float y1 = (v-min_v)/step;
        //     float r = 1.414f * model[3]/step;
        //     float rx = r * model[5] / sqrt(model[4]*model[4]+model[5]*model[5]);
        //     float ry = r * model[6] / sqrt(model[4]*model[4]+model[6]*model[6]);
        //     float rotation = 90 - atan(model[6]/model[5]) * 180 / PI;
        //     // std::cout << model[0] << " " << model[1] << " " << model[2] << " " << model[3] << " " 
        //     //           << model[4] << " " << model[5] << " " << model[6] << std::endl;
        //     // std::cout << "x: " << x1 << " y: " << y1 << " r: " << r << std::endl;
        //     // std::cout << "rx: " << rx << " ry: " << ry << " rotation: " << rotation << std::endl;
        //     cv::ellipse(image, cv::Point(x1, y1), cv::Size(rx, ry), 
        //                 rotation, 0, 360, cv::Scalar(0, 0, 255, 255), 1, cv::LINE_AA);
        // }

    } else if (m_application == "sphere") {
        // for (const auto& model : m_bestModelCoefficients) {
        //     // origin(x,y,z) R
        //     float u = projection(model[1], model[0]);
        //     float v = projection(model[2], model[0]);
        //     float x1 = (u-min_u)/step;
        //     float y1 = (v-min_v)/step;
        //     float r = 1.414f * model[3]/step;
        //     std::cout << model[0] << " " << model[1] << " " << model[2] << " " << model[3] << std::endl;
        //     std::cout << "x: " << x1 << " y: " << y1 << " r: " << r << std::endl;   
        //     cv::circle(image, cv::Point(x1, y1), r, cv::Scalar(0, 0, 255, 255), -1, cv::LINE_AA);
        // }

    } else if (m_application == "cylinder") {
        std::cout << "Here is some bug..." << std::endl;

    } else {
        std::cerr << "Invalid application" << std::endl;
        abort();
    }

    // draw the x, y, and z axis
    // cv::line(image, cv::Point(0, height - 1), cv::Point((width - 1)/3, 2*(height-1)/3), cv::Scalar(64, 64, 64, 255), 1, cv::LINE_AA);
    // cv::line(image, cv::Point((width - 1)/3, 0), cv::Point((width - 1)/3, 2*(height-1)/3), cv::Scalar(64, 64, 64, 255), 1, cv::LINE_AA);
    // cv::line(image, cv::Point(width - 1, 2*(height - 1)/3), cv::Point((width - 1)/3, 2*(height-1)/3), cv::Scalar(64, 64, 64, 255), 1, cv::LINE_AA);

    for (const auto& point : projectedPoints) {
        int x = (point.x - min_u) / step;
        int y = (point.y - min_v) / step;
        cv::circle(image, cv::Point(x, y), 1, cv::Scalar(0, 0, 0), -1, cv::LINE_AA);
    }

    return image;
}