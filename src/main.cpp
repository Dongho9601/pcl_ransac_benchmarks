#include "main.hpp"

int main(int argc, char* argv[])
{
    if (argc != 5) {
        cerr << "Usage: [dimension] [application] [path/to/pcd] [CPU/GPU]" << endl;
        cerr << "2D applications: line, circle" << endl;
        cerr << "3D applications: line, plane, circle, cylinder, sphere" << endl;
        return 0;
    }
    
    const std::string application = argv[2];
    const std::string pclFilePath = argv[3];
    const std::string device = argv[4];
    
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

    Timer timer;
    timer.clear();

    if (dimension == 2) {
        std::cout << "2D point cloud" << std::endl;

        timer.start();
        Fitter2D fitter(application, device);
        fitter.run(cloud);
        std::cout << "Execution time: " << timer.stop() << std::endl;

        cv::Mat image = fitter.draw2DImage(cloud);
        cv::imwrite("LineFitterImage.png", image);

    } else {
        std::cout << "3D" << std::endl;

    }

    return 0;
}