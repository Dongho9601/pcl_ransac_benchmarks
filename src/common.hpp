#pragma once
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>

using PointCloudType = pcl::PointXYZ;
using PointCloud = pcl::PointCloud<PointCloudType>;
using PointCloudPtr = PointCloud::Ptr;
using Coefficients = pcl::ModelCoefficients;
using CoefficientsPtr = pcl::ModelCoefficients::Ptr;
// using LinesFitting = perception::LinesFitting<PointCloudType, 2>;

using namespace std;