#include "common.hpp"
#include <random>

#include <pcl/filters/random_sample.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/sac_model_circle.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/random_sample.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 32 * 8
#define WARP_PER_BLOCK 8

#define DIMENSION 2
#define PI 3.141592f

class Fitter2D {
    public:
        Fitter2D(){};
        Fitter2D(const std::string app_, const std::string device_): m_application(app_), m_device(device_) {
            if (m_application == "line") {
                m_numRequiredPoints = 2;

            } else if (m_application == "circle") {
                m_numRequiredPoints = 3;

            } else {
                std::cerr << "Invalid application" << std::endl;
                abort();
            }
        };
        ~Fitter2D(){};

        void run(const PointCloudPtr& cloud);
        void getBestModelCoefficients(Eigen::VectorXf& bestModelCoefficients_) {
             m_bestModelCoefficients.push_back(bestModelCoefficients_);
        };
        cv::Mat draw2DImage(const PointCloudPtr& cloud,
                            const float step = 0.008,
                            const int defaultWidth = 500, 
                            const int defaultHeight = 500);

        template <typename modelType> void runFitting(PointCloudPtr& cloudCopy, modelType& model);
        void runFittingWithCUDA(PointCloudPtr& cloudCopy);

    private:
        const std::string m_application;
        const std::string m_device;

        int m_numRequiredPoints = 2;
        int m_maxIterations = 1000;
        float m_threshold = 0.99;
        float m_delta = 0.006;
        float m_remainingPointsRatio = 1;

        std::vector<Eigen::VectorXf> m_bestModelCoefficients;
};
