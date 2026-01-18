#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>
#include <vector>
#include <tuple>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <string>

#define MAX_ITER 1500

#define a 0.01
struct Point {
    double x;
    double y;
};
struct Window {
    int width;
    int height;
};
struct RGB { unsigned char r, g, b; };
__global__ void addKernel(cuDoubleComplex* z, int* n, const cuDoubleComplex* c, int max)
{
    //int i = threadIdx.x;
    //c[i] = a[i] + b[i];
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int iterations = -1;
    cuDoubleComplex z_local = z[p];
    cuDoubleComplex c_local = c[p];

    for (int i = 0; i < max; i++) {
        z_local = cuCadd(cuCmul(z_local, z_local), c_local);

        double mag_sq = z_local.x * z_local.x + z_local.y * z_local.y;
        if (mag_sq > 4.0) {
            iterations = i;
            break;
        }
    }

    z[p] = z_local;
    n[p] = iterations;
}
__global__ void fillValue(int* arr, int size, int val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        arr[idx] = val;
    }
}
__global__
void colorKernel(
    const int* n,
    uchar3* rgb,
    int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    int iter = n[i];

    // Use likely/unlikely hints for better branch prediction
    if (iter == -1) {
        rgb[i] = make_uchar3(0, 0, 0);
        return;
    }

    // Use float instead of double for faster computation
    // Pre-compute constant multiplier
    float normalizedIter = (float)2*(std::log(a*iter+1)/a) * 0.1f;

    // Use fast single-precision intrinsics
    float sinVal = sinf(normalizedIter);
    float cosVal = cosf(normalizedIter);

    // Compute colors with single multiplication and addition
    unsigned char r = (unsigned char)(sinVal * 127.0f + 128.0f);
    unsigned char g = (unsigned char)(cosVal * 127.0f + 128.0f);
    unsigned char b = 255;

    rgb[i] = make_uchar3(r, g, b);
}
__global__ void coordsKernel(cuDoubleComplex* coords,
    int width,
    int height,
    double centerX,
    double centerY,
    double zoom
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;
    
    int x = idx % width;
    int y = idx / width;
    
    double real = fma((double)(x - width / 2), 1.0 / (zoom * width), centerX);
    double imag = fma((double)(y - height / 2), 1.0 / (zoom * height), centerY);
    
    coords[idx] = make_cuDoubleComplex(real, imag);
}
class CudaFractalGenerator {

    cuDoubleComplex* d_z = nullptr;
    cuDoubleComplex* d_c = nullptr;
    int* d_n = nullptr;
    uchar3* d_rgb = nullptr;
    int size = 0;
    int width = 1024;
    int height = 1024;
    int blockSize = 256;
    int gridSize = 0;
    cudaStream_t stream = nullptr;

public:
    void init(int width, int height) {
        this->size = width * height;
        this->width = width;
        this->height = height;
        this->gridSize = (size + blockSize - 1) / blockSize;

        cudaMalloc((void**)&d_z, size * sizeof(cuDoubleComplex));
        cudaMalloc((void**)&d_c, size * sizeof(cuDoubleComplex));
        cudaMalloc((void**)&d_n, size * sizeof(int));
        cudaMalloc((void**)&d_rgb, size * sizeof(uchar3));

        cudaStreamCreate(&stream);
    }

    void run(std::vector<uchar3>& rgb, Point center, Window window, double zoom, int max) {
        // Generate coordinates
        coordsKernel << <gridSize, blockSize, 0, stream >> > (
            d_c,
            window.width,
            window.height,
            center.x,
            center.y,
            zoom
            );

        // Initialize z to zero and n to -1 in parallel
        cudaMemsetAsync(d_z, 0, size * sizeof(cuDoubleComplex), stream);
        fillValue << <gridSize, blockSize, 0, stream >> > (d_n, size, -1);

        // Compute Mandelbrot iterations
        addKernel << <gridSize, blockSize, 0, stream >> > (d_z, d_n, d_c, max);

        // Generate colors
        colorKernel << <gridSize, blockSize, 0, stream >> > (
            d_n,
            d_rgb,
            size
            );

        // Async copy back to host
        cudaMemcpyAsync(rgb.data(), d_rgb, size * sizeof(uchar3), cudaMemcpyDeviceToHost, stream);

        // Wait for all operations to complete
        cudaStreamSynchronize(stream);
    }

    void cleanup() {
        if (stream) cudaStreamDestroy(stream);
        cudaFree(d_z);
        cudaFree(d_c);
        cudaFree(d_n);
        cudaFree(d_rgb);
    }

    ~CudaFractalGenerator() {
        cleanup();
    }
};

int main()
{
    const int width = 1024;
    const int height = 1024;

    const int arraySize = width * height;
    //Point center = { -1.74528, 0.003331 };
    Point center = { -1.7452751580045,-0.0033287424761 };
	Window window = { width, height };
    //Point center = { -0.1528467399413 , -1.0396951458385 };
    double zoom = 0.5;

    std::vector<cuDoubleComplex> z(arraySize, make_cuDoubleComplex(0.0, 0.0));
    std::vector<int> n(arraySize, -1.0);
    //uchar3* rgb;
	std::vector<uchar3> rgb(arraySize);
    //cudaMallocHost(&rgb, arraySize * sizeof(uchar3));
    // ... use rgb ...
    //cudaFreeHost(rgb);

    cv::VideoWriter writer("fractal_video2.mp4",
        cv::VideoWriter::fourcc('a', 'v', 'c', '1'), 60, cv::Size(width, height));

    CudaFractalGenerator gen;
    gen.init(width, height);
    int maxIter = 100;
    // END CONSTANTS
    // START LOOP

    std::vector<cuDoubleComplex> coords;
    for (int frame = 0; frame < 1080 ; ++frame) {
        zoom *= 1.03;
        // init coordinates

        auto start = std::chrono::high_resolution_clock::now();
        //prepareCoordinates(coords, width, height, center.x, center.y, zoom);
		maxIter = std::min(maxIter + 3, MAX_ITER);
        gen.run(rgb, center, window, zoom, maxIter );



        cv::Mat image(height, width, CV_8UC3);

        // Direct memcpy from rgb vector data to image data
        memcpy(image.data, rgb.data(), width * height * sizeof(uchar3));
		cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        printf("Frame %d completed with maxIter %d zoom %.2f in %.5f seconds\n", frame, maxIter, zoom, elapsed);

        // Write the frame
        writer.write(image);
        cv::imshow("Preview", image);
        int key = cv::waitKey(1);
        if (key == 27) break;

        //reset
        //std::fill(z.begin(), z.end(), make_cuDoubleComplex(0.0,0.0));
        //std::fill(n.begin(), n.end(), -1.0);
    }

    // CLEANUP/FINISH
    writer.release();
    gen.cleanup();
    return 0;
}



