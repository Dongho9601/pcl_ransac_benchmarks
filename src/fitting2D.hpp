#include "common.hpp"

#include <pcl/filters/random_sample.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/sac_model_circle.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/random_sample.h>

// #include <pcl/cuda/sample_consensus/sac_model.h>
// #include <pcl/cuda/sample_consensus/ransac.h>

// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>
// #include <thrust/shuffle.h>
// #include <thrust/random.h>

#include <random>

#define WARP_SIZE 32
#define BLOCK_SIZE 32 * 8
#define WARP_PER_BLOCK 8

/* Pseudocode of RANSAC
https://en.wikipedia.org/wiki/Random_sample_consensus

Given:
    data – A set of observations.
    model – A model to explain the observed data points.
    n – The minimum number of data points required to estimate the model parameters.
    k – The maximum number of iterations allowed in the algorithm.
    t – A threshold value to determine data points that are fit well by the model (inlier).
    d – The number of close data points (inliers) required to assert that the model fits well to the data.

Return:
    bestFit – The model parameters which may best fit the data (or null if no good model is found).

iterations = 0
bestFit = null
bestErr = something really large // This parameter is used to sharpen the model parameters to the best data fitting as iterations goes on.

while iterations < k do
    maybeInliers := n randomly selected values from data
    maybeModel := model parameters fitted to maybeInliers
    confirmedInliers := empty set
    for every point in data do
        if point fits maybeModel with an error smaller than t then
             add point to confirmedInliers
        end if
    end for
    if the number of elements in confirmedInliers is > d then
        // This implies that we may have found a good model.
        // Now test how good it is.
        betterModel := model parameters fitted to all the points in confirmedInliers
        thisErr := a measure of how well betterModel fits these points
        if thisErr < bestErr then
            bestFit := betterModel
            bestErr := thisErr
        end if
    end if
    increment iterations
end while
return bestFit
*/

/* References
https://github.com/xmba15/ransac_lines_fitting_gpu
https://github.com/leomariga/pyRANSAC-3D
https://github.com/kzampog/cilantro/tree/master
*/

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

        void pointsToVector(float& pointsV, const PointCloudPtr& cloud);
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
