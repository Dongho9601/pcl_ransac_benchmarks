#include "common.hpp"
#include <pcl/filters/random_sample.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/filters/extract_indices.h>
// #include </__w/1/s/cuda/sample_consensus/include/pcl/cuda/sample_consensus/sac_model.h>

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
        Fitter2D(const std::string app_, const std::string device_): application(app_), device(device_) {
            if (application == "line") {
                numRequiredPoints = 2;
                maxIterations = 1000;
                threshold = 0.99;
                delta = 0.006;
                remainingPointsRatio = 1;

            } else if (application == "plane") {
                numRequiredPoints = 3;

            } else if (application == "circle") {
                numRequiredPoints = 3;

            } else {
                std::cerr << "Invalid application" << std::endl;
                abort();
            }
        };
        ~Fitter2D(){};

        void run(const PointCloudPtr& cloud);
        void getBestModelCoefficients(Eigen::VectorXf& bestModelCoefficients_) {
             bestModelCoefficients.push_back(bestModelCoefficients_);
        };
        cv::Mat draw2DImage(const PointCloudPtr& cloud,
                            const float xStep = 0.008, const float yStep = 0.008, 
                            const int defaultWidth = 500, const int defaultHeight = 500);

        void lineFitterRun(PointCloudPtr& cloudCopy);
        void lineFitterRunCUDA(PointCloudPtr& cloudCopy);

    private:
        const std::string application;
        const std::string device;

        int numRequiredPoints = 2;
        int maxIterations = 100;
        float threshold = 0.99;
        float delta = 0.006;
        float remainingPointsRatio = 0.8;

        vector<Eigen::VectorXf> bestModelCoefficients;
};
