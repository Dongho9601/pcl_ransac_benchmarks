#include "main.hpp"

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

using namespace std;

class LineFitter {
    public:
        LineFitter(){};
        LineFitter(const int maxIterations_, const float threshold_, const float delta_):
            maxIterations(maxIterations_), threshold(threshold_), delta(delta_) {};
        ~LineFitter(){};

        void run(const PointCloudPtr& cloud);
        void getBestModelCoefficients(Eigen::VectorXf& bestModelCoefficients_) {
             bestModelCoefficients.push_back(bestModelCoefficients_);
        };
        cv::Mat draw2DImage(const PointCloudPtr& cloud,
                            const float xStep = 0.008, const float yStep = 0.008, 
                            const int defaultWidth = 500, const int defaultHeight = 500);

    private:
        const int numRequiredPoints = 2;
        int maxIterations = 100;
        float threshold = 0.99;
        float delta = 0.006;

        vector<Eigen::VectorXf> bestModelCoefficients;
};

void LineFitter::run(const PointCloudPtr& cloud) {
    // compute RANSAC model
    pcl::SampleConsensusModelLine<PointCloudType>::Ptr model(new pcl::SampleConsensusModelLine<PointCloudType>(cloud));
    pcl::RandomSampleConsensus<PointCloudType> ransac(model);
    ransac.setMaxIterations(maxIterations);
    ransac.setProbability(threshold);
    ransac.setDistanceThreshold(delta);
    ransac.computeModel();

    // print inliners
    // PointCloudPtr inlinerPoints(new PointCloud);
    // pcl::copyPointCloud(*cloud, inliners, *inlinerPoints);
    // std::cout << "inlinerPoints: " << inlinerPoints->points.size() << std::endl;
    // for (int i = 0; i < inlinerPoints->points.size(); i++) {
    //     std::cout << inlinerPoints->points[i].x << ", " << inlinerPoints->points[i].y << ", " << inlinerPoints->points[i].z << std::endl;
    // }

    // print the best model
    Eigen::VectorXf modelCoefficients;
    ransac.getModelCoefficients(modelCoefficients);
    getBestModelCoefficients(modelCoefficients);
    // std::cout << "modelCoefficients: \n" << modelCoefficients << std::endl;

    // TODO: remove inliners and do again for the rest of points
}

cv::Mat LineFitter::draw2DImage(const PointCloudPtr& inCloud,
                                const float xStep, const float yStep, 
                                const int defaultWidth, const int defaultHeight)
{
    cv::Scalar bgColor = cv::Scalar(167, 167, 167);
    if (inCloud->size() < 2) {
        return cv::Mat(defaultHeight, defaultWidth, CV_8UC3, bgColor);
    }

    pcl::PointXYZ minP, maxP;
    pcl::getMinMax3D<pcl::PointXYZ>(*inCloud, minP, maxP);

    int height = (maxP.y - minP.y) / yStep;
    int width = (maxP.x - minP.x) / xStep;

    cv::Mat image(height, width, CV_8UC3, bgColor);
    std::cout << "image size: " << image.size() << std::endl;
    for (const auto& point : inCloud->points) {
        int x = (point.x - minP.x) / xStep;
        int y = (point.y - minP.y) / yStep;
        image.ptr<cv::Vec3b>(y)[x] = cv::Vec3b(0, 0, 255);
        cv::circle(image, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
    }

    // add line "P = (model[0],model[1]) + t(model[3],model[4])"" in the bestModelCoefficients
    for (const auto& model : bestModelCoefficients) {
        float minT = std::abs((minP.x-model[0])/model[3]) > abs((minP.y-model[1])/model[4]) ?
                     (minP.x-model[0])/model[3] : (minP.y-model[1])/model[4];
        float maxT = std::abs((maxP.x-model[0])/model[3]) > abs((maxP.y-model[1])/model[4]) ?
                     (maxP.x-model[0])/model[3] : (maxP.y-model[1])/model[4] ;
        float x1 = (model[0]+minT*model[3]-minP.x)/xStep;
        float y1 = (model[1]+minT*model[4]-minP.y)/yStep;
        float x2 = (model[0]+maxT*model[3]-minP.x)/xStep;
        float y2 = (model[1]+maxT*model[4]-minP.y)/yStep;
        cv::line(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
    }

    return image;
}

int main(int argc, char* argv[])
{
    if (argc != 5) {
        cerr << "Usage: [dimension] [app] [path/to/pcd] [CPU/GPU]" << endl;
        cerr << "2D apps: point, line, plane, circle" << endl;
        cerr << "3D apps: point, line, plane, circle, cylinder, sphere" << endl;
        return 0;
    }
    
    const std::string pclFilePath = argv[3];
    
    // load pcl file
    std::cout << "pclFilePath: " << pclFilePath << std::endl;
    PointCloudPtr cloud (new PointCloud);
    if (pcl::io::loadPCDFile(pclFilePath, *cloud) == -1) {
        cerr << "Failed to load pcl file" << endl;
        return 0;
    }

    int dimension = atoi(argv[1]);
    if (dimension != 2 && dimension != 3) {
        std::cerr << "Invalid dimension" << std::endl;
        return 0;
    }

    // fitting
    if (dimension == 2) {
        std::cout << "2D point cloud" << std::endl;
        LineFitter lineFitter(1000, 0.99, 0.006);
        lineFitter.run(cloud);
        cv::Mat image = lineFitter.draw2DImage(cloud);
        cv::imwrite("2Dimage.png", image);

    } else {
        std::cout << "3D" << std::endl;

    }

    return 0;
}