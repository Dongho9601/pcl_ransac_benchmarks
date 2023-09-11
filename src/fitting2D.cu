#include "fitting2D.hpp"

__global__ 
void lineFitting2DCUDA(float* pointsArr, int* inlinerCounts, int pointsNum, int maxIterations, float delta) {
    // warps are models, so threads search the points divided by 32
    int warpIdx = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int laneIdx = threadIdx.x % WARP_SIZE;

    // model generation
    // the first point is the leader thread's warpIdx % pointsNum
    // the second point is the leader thread's warpIdx % pointsNum + warpIdx / pointsNum +1
    int firstPointIdx = warpIdx % pointsNum;
    int secondPointIdx = firstPointIdx + warpIdx / pointsNum + 1;
    if (secondPointIdx >= pointsNum)
        secondPointIdx %= pointsNum;

    float Ox = pointsArr[firstPointIdx * DIMENSION];
    float Oy = pointsArr[firstPointIdx * DIMENSION + 1];
    float Rx = pointsArr[secondPointIdx * DIMENSION] - Ox;
    float Ry = pointsArr[secondPointIdx * DIMENSION + 1] - Oy;

    // each thread calculates the number of inliners and accumulates them at the end
    int counter = 0;
#pragma unroll
    for (int i = 0; i < pointsNum / WARP_SIZE + 1; i++) {
        int pointIdx = i * WARP_SIZE + laneIdx ;
        if (pointIdx >= pointsNum) break;
        float Qx = pointsArr[pointIdx * DIMENSION];
        float Qy = pointsArr[pointIdx * DIMENSION + 1];
        float distance = abs(Rx * (Qy - Oy) - Ry * (Qx - Ox)) / sqrt(Rx * Rx + Ry * Ry);
        if (distance < delta) {
            counter++;
        }
    }

    // accumulate the inliner counts within a warp
    for (int i = 16; i > 0; i /= 2)
        counter += __shfl_down_sync(0xffffffff, counter, i);
    
    // write the inliner counts to the global memory
    if (laneIdx == 0)
        inlinerCounts[warpIdx] = counter;
}

__global__ 
void circleFitting2DCUDA(float* pointsArr, int* inlinerCounts, int pointsNum, int maxIterations, float delta) {
    int warpIdx = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int laneIdx = (blockIdx.x * blockDim.x + threadIdx.x) % WARP_SIZE;

    int firstPointIdx = warpIdx % pointsNum;
    int secondPointIdx = firstPointIdx + warpIdx / pointsNum + 1;
    if (secondPointIdx >= pointsNum)
        secondPointIdx %= pointsNum;
    int thridPointIdx = secondPointIdx + warpIdx / pointsNum + 1;
    if (thridPointIdx >= pointsNum)
        thridPointIdx %= pointsNum;

    // calculate the center of the circle
    float Ox = (pointsArr[firstPointIdx * DIMENSION] + pointsArr[secondPointIdx * DIMENSION] + pointsArr[thridPointIdx * DIMENSION]) / 3;
    float Oy = (pointsArr[firstPointIdx * DIMENSION + 1] + pointsArr[secondPointIdx * DIMENSION + 1] + pointsArr[thridPointIdx * DIMENSION + 1]) / 3;
    float R = sqrt((pointsArr[firstPointIdx * DIMENSION] - Ox) * (pointsArr[firstPointIdx * DIMENSION] - Ox) 
                   + (pointsArr[firstPointIdx * DIMENSION + 1] - Oy) * (pointsArr[firstPointIdx * DIMENSION + 1] - Oy));

    // each thread calculates the number of inliners and accumulates them at the end
    int counter = 0;
#pragma unroll
    for (int i = 0; i < pointsNum / WARP_SIZE + 1; i++) {
        int pointIdx = i * WARP_SIZE + laneIdx;
        if (pointIdx >= pointsNum) break;
        float Qx = pointsArr[pointIdx * DIMENSION];
        float Qy = pointsArr[pointIdx * DIMENSION + 1];
        float distance = abs(sqrt((Qx - Ox) * (Qx - Ox) + (Qy - Oy) * (Qy - Oy)) - R);
        if (distance < delta) {
            counter++;
        }
    }

    // accumulate the inliner counts within a warp
    for (int i = 16; i > 0; i /= 2)
        counter += __shfl_down_sync(0xffffffff, counter, i);
    
    // write the inliner counts to the global memory
    if (laneIdx == 0)
        inlinerCounts[warpIdx] = counter;

}

void Fitter2D::runFittingWithCUDA(PointCloudPtr& cloudCopy) {
    // point to array
    float* pointsArr = new float[cloudCopy->points.size() * DIMENSION];
    for (int i = 0; i < cloudCopy->points.size(); i++) {
        pointsArr[DIMENSION * i] = (float)cloudCopy->points[i].x;
        pointsArr[DIMENSION * i + 1] = (float)cloudCopy->points[i].y;
    }

    // shuffle the pointsArr with std::suffle
    std::random_device rd;
    std::mt19937 g(rd());
    for (auto i=cloudCopy->points.size()-1; i>0; --i) {
        std::uniform_int_distribution<decltype(i)> d(0,i);
        std::swap (pointsArr[i], pointsArr[d(g)]);
        std::swap (pointsArr[i+1], pointsArr[d(g)+1]);
    }

    Timer timer;
    timer.clear();
    timer.start();

    // upload the pointsArr to GPUs
    float* pointsArr_d;
    cudaMalloc((void**)&pointsArr_d, cloudCopy->points.size() * DIMENSION * sizeof(float));
    cudaMemcpy(pointsArr_d, pointsArr, cloudCopy->points.size() * DIMENSION * sizeof(float), cudaMemcpyHostToDevice);

    // setup inliner counts
    int* inlinerCounts = new int[m_maxIterations];
    int* inlinerCounts_d;
    cudaMalloc((void**)&inlinerCounts_d, m_maxIterations * sizeof(int));
    cudaMemset(inlinerCounts_d, 0, m_maxIterations * sizeof(int));

    std::cout << "GPU malloc time: " << timer.stop() << std::endl;

    // warp numbers are model numbers
    // thread block size is 256, fixed = 8 warps
    if (m_application == "line") {
        lineFitting2DCUDA<<<m_maxIterations / WARP_PER_BLOCK, BLOCK_SIZE>>>(pointsArr_d, inlinerCounts_d, cloudCopy->points.size(), m_maxIterations, m_delta);
    }
    else if (m_application == "circle") {
        circleFitting2DCUDA<<<m_maxIterations / WARP_PER_BLOCK, BLOCK_SIZE>>>(pointsArr_d, inlinerCounts_d, cloudCopy->points.size(), m_maxIterations, m_delta);
    }

    // download the inliner counts
    cudaMemcpy(inlinerCounts, inlinerCounts_d, m_maxIterations * sizeof(int), cudaMemcpyDeviceToHost);

    // get the best model
    int bestModelIdx = 0;
    int bestModelInlinerCount = 0;
    for (int i = 0; i < m_maxIterations; i++) {
        if (inlinerCounts[i] > bestModelInlinerCount) {
            bestModelIdx = i;
            bestModelInlinerCount = inlinerCounts[i];
        }

        // probability of success
        // float p = m_threshold;
        // float w = (float)inlinerCounts[i] / (float)m_maxIterations;
        // float N = log(1 - p) / log(1 - pow(w, m_numRequiredPoints));
        // if (N < m_maxIterations) {
        //     m_maxIterations = (int) N;
        // }
    }

    // print the best model
    std::cout << "best model: " << bestModelIdx << std::endl;
    std::cout << "best model inliner count: " << bestModelInlinerCount << std::endl;

    // print the coordinates of the best model
    if (m_application == "line") {
        int firstPointIdx = bestModelIdx % cloudCopy->points.size();
        int secondPointIdx = firstPointIdx + bestModelIdx / cloudCopy->points.size() + 1;
        if (secondPointIdx >= cloudCopy->points.size()) {
            secondPointIdx -= cloudCopy->points.size();
        }
        float Ox = pointsArr[firstPointIdx * DIMENSION];
        float Oy = pointsArr[firstPointIdx * DIMENSION + 1];
        float Rx = pointsArr[secondPointIdx * DIMENSION] - Ox;
        float Ry = pointsArr[secondPointIdx * DIMENSION + 1] - Oy;

        Eigen::VectorXf model_(6);
        model_ << Ox, Oy, 0, Rx, Ry, 0;
        getBestModelCoefficients(model_);
    }
    else if (m_application == "circle") {
        int firstPointIdx = bestModelIdx % cloudCopy->points.size();
        int secondPointIdx = firstPointIdx + bestModelIdx / cloudCopy->points.size() + 1;
        if (secondPointIdx >= cloudCopy->points.size()) {
            secondPointIdx -= cloudCopy->points.size();
        }
        int thridPointIdx = secondPointIdx + 1;
        if (thridPointIdx >= cloudCopy->points.size()) {
            thridPointIdx -= cloudCopy->points.size();
        }
        float Ox = (pointsArr[firstPointIdx * DIMENSION] + pointsArr[secondPointIdx * DIMENSION] + pointsArr[thridPointIdx * DIMENSION]) / 3;
        float Oy = (pointsArr[firstPointIdx * DIMENSION + 1] + pointsArr[secondPointIdx * DIMENSION + 1] + pointsArr[thridPointIdx * DIMENSION + 1]) / 3;
        float R = sqrt((pointsArr[firstPointIdx * DIMENSION] - Ox) * (pointsArr[firstPointIdx * DIMENSION] - Ox) 
                  + (pointsArr[firstPointIdx * DIMENSION + 1] - Oy) * (pointsArr[firstPointIdx * DIMENSION + 1] - Oy));

        Eigen::VectorXf model_(3);
        model_ << Ox, Oy, R;
        getBestModelCoefficients(model_);
    }    

    // free the memory
    delete[] pointsArr;
    delete[] inlinerCounts;
    cudaFree(pointsArr_d);
    cudaFree(inlinerCounts_d);
}