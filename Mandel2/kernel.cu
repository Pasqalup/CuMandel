#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>
#include <vector>
#include "stb_image_write.h"
#include <tuple>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <format>
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
    
    // Precompute scale factors (constant for all threads in the same kernel launch)
    double scaleX = 1.0 / (zoom * width);
    double scaleY = 1.0 / (zoom * height);
    
    // Precompute half dimensions
    double halfWidth = width * 0.5;
    double halfHeight = height * 0.5;
    
    int x = idx % width;
    int y = idx / width;
    
    // Map pixel -> complex plane with reduced operations
    double real = centerX + (x - halfWidth) * scaleX;
    double imag = centerY + (y - halfHeight) * scaleY;
    
    coords[idx] = make_cuDoubleComplex(real, imag);
}
//std::tuple<unsigned char, unsigned char, unsigned char> getColor(int iter, int maxIter) {
//    if (iter == maxIter) {
//        return { 0, 0, 0 };  // Black for points inside the set
//    }
//    // Smooth coloring example:
//    double a = 0.01;
//    double normalizedIter = 2.0* (std::log(1.0 + a * iter) / a);
//    double t = static_cast<double>(normalizedIter) * 0.1 + 0.1;
//    unsigned char r = static_cast<unsigned char>(sin(t) * 127 + 128);
//    unsigned char g = static_cast<unsigned char>(cos(t) * 127 + 128);
//    unsigned char b = static_cast<unsigned char>(255);
//    return { r, g, b };
//}
RGB getColor2(int iter) {
    double normalizedIter = (2.0 * (std::log(1.0 + a * iter) / a)) * 0.1 + 0.1;
    return{
        unsigned char(sin(normalizedIter) * 127 + 128),
        unsigned char(cos(normalizedIter) * 127 + 128),
        255
    };
}
void prepareCoordinates(
    std::vector<cuDoubleComplex>& coords,
    int width,
    int height,
    double centerX,
    double centerY,
    double zoom
) {
    coords.resize(width * height);

    double scale = 1.0 / zoom;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {

            int idx = y * width + x;

            // Map pixel -> complex plane
            double real = centerX
                + (x - width / 2.0) * scale / width;

            double imag = centerY
                + (y - height / 2.0) * scale / height;

            coords[idx] = make_cuDoubleComplex(real, imag);
        }
    }
}
void saveFractalPNG(const std::vector<unsigned char>& rgbData, int width, int height,int i) {
	std::string filename = "frame" + std::to_string(i) + ".png";
    if (!stbi_write_png("fractal%d.png", width, height, 3, rgbData.data(), width * 3)) {
        std::cerr << "Failed to write image\n";
    }
}
class CudaFractalGenerator {

    cuDoubleComplex* d_z = nullptr;
    cuDoubleComplex* d_c = nullptr;
    int* d_n = nullptr;
	uchar3* d_rgb = nullptr;
    int size = 0;
    int width = 1024;
    int height = 1024;
public:
    void init(int width, int height) {
        this->size = width * height;
		this->width = width;
		this->height = height;
        cudaMalloc((void**)&d_z, size * sizeof(cuDoubleComplex));
        cudaMalloc((void**)&d_c, size * sizeof(cuDoubleComplex));
        cudaMalloc((void**)&d_n, size * sizeof(int));
		cudaMalloc((void**)&d_rgb, size * sizeof(uchar3));
    }
    void run(std::vector<uchar3>& rgb, Point center, Window window, double zoom, int max) {
        //cudaMemcpy(d_c, c.data(), size * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        // 
        //cudaMemcpy(d_z, z.data(), size * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
		cudaMemset(d_z, 0, size * sizeof(cuDoubleComplex));
        //cudaMemcpy(d_n, n.data(), size * sizeof(int), cudaMemcpyHostToDevice);
        int blockSize = 256; // or 512
        int gridSize = (size + blockSize - 1) / blockSize;
        coordsKernel<<<gridSize, blockSize>>>(
            d_c,
            window.width,
            window.height,
            center.x,
            center.y,
            zoom
			);
        fillValue << <gridSize, blockSize >> > (d_n, size, -1.0);


        addKernel << <gridSize, blockSize >> > (d_z, d_n, d_c, max);
        colorKernel<<<gridSize, blockSize>>>(
            d_n,
            d_rgb,
            size
			);
        cudaDeviceSynchronize();
        cudaMemcpy(rgb.data(), d_rgb, size * sizeof(uchar3), cudaMemcpyDeviceToHost);

    }
    void cleanup() {
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
	std::vector<uchar3> rgb(arraySize);

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



