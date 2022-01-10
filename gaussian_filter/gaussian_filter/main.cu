#include <iostream>
#include <string>
#include <fstream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define PIXEL double

typedef struct _BITMAP{
    int height;
    int width;
    PIXEL* bytes = NULL;
} BITMAP, *PBITMAP;

void destroyBitmap(PBITMAP bitmap) {
    if (bitmap != NULL) {
        if (bitmap->bytes != NULL) {
            cudaFree(&(bitmap->bytes));
        }
        cudaFree(&bitmap);
    }
}

__device__
PIXEL _max(PIXEL x, PIXEL y) {
    if (x > y) {
        return x;
    }
    return y;
}

__device__
PIXEL _min(PIXEL x, PIXEL y) {
    if (x < y) {
        return x;
    }
    return y;
}

__device__
PIXEL applyFilter(PBITMAP image, PBITMAP kernel, int offset)
{
    int row = offset / image->width;
    int col = offset % image->width;
    PIXEL returned = 0;
    int kernelDimX = kernel->width / 2;
    int kernelDimY = kernel->height / 2;
    int startY = row - kernelDimY;
    int startX = col - kernelDimX;

    for (int i = 0; i <= kernel->height; i++) 
    {
        for (int j = 0; j <= kernel->width; j++)
        {
            int newY = _max(0, _min(startY + i, image->height - 1));
            int newX = _max(0, _min(startX + j, image->width - 1));
            int position = newY * image->width + newX;
            returned += image->bytes[position] * kernel->bytes[i * kernel->width + j];
        }
    }

    return returned;
}

__global__
void gaussian_filter(PBITMAP image, PBITMAP kernel, PBITMAP result) 
{
    int stride = blockDim.x * gridDim.x;
    int rest = (image->width * image->height) % stride, cat = image->width * image->height / stride;
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int start = cat * threadId + _min(threadId, rest);
    int extra = 0;

    if (threadId < (image->width * image->height) % stride) {
        extra = 1;
    }

    for (int i = 0; i < cat + extra && start + i < image->height * image->width; i++) {
        int offset = start + i;
        PIXEL x = applyFilter(image, kernel, offset);
        result->bytes[offset] = x;
    }
}

int readMatrix(PBITMAP& matrix, std::ifstream& fin) {
    int error = cudaMallocManaged(&matrix, sizeof(BITMAP));

    if (error != cudaSuccess) {
        return -1;
    }
    fin >> matrix->height >> matrix->width;

    int size = matrix->height * matrix->width;
    error = cudaMallocManaged(&(matrix->bytes), size * sizeof(PIXEL));
    if (error != cudaSuccess) {
        return -2;
    }

    for (int i = 0; i < size; i++) {
        fin >> matrix->bytes[i];
    }

    return 0;
}

int main(int argc, char** argv)
{
    int status = 0;
    std::string filename;
    std::ifstream fin;
    std::ofstream fout;
    PBITMAP image = NULL, kernel = NULL, result = NULL;

    if (argc < 3) {
        std::cout << "Usage: program filename no_threads\n";
        return -1;
    }
    filename = argv[1];

    fin.open(filename);
    if (!fin.is_open()) {
        std::cout << "Couldn't open file " + filename + "\n";
        goto cleanup;
    }

    if ((status = readMatrix(image, fin)) < 0) {
        std::cout << "Error reading image\n";
        goto cleanup;
    }

    if ((status = readMatrix(kernel, fin)) < 0) {
        std::cout << "Error reading kernel\n";
        goto cleanup;
    }

    if (cudaMallocManaged(&result, sizeof(PBITMAP)) != cudaSuccess) {
        std::cout << "Error allocating result image\n";
        goto cleanup;
    }
    if (cudaMallocManaged(&result->bytes, sizeof(PIXEL) * image->height * image->width) != cudaSuccess) {
        std::cout << "Error allocating result image\n";
        goto cleanup;
    }
    result->height = image->height;
    result->width = image->width;

    int noThreads = atoi(argv[2]);
    int noBlocks = (image->height * image->width + noThreads - 1) / noThreads;

    const clock_t beginTime = clock();

    gaussian_filter<<<noBlocks, noThreads>>>(image, kernel, result);
    cudaDeviceSynchronize();

    std::cout << 1000.0 * (float(clock() - beginTime) / CLOCKS_PER_SEC);

    fout.open("output.txt");
    if (!fout.is_open()) {
        std::cout << "Couldn't open/create output file\n";
        goto cleanup;
    }

    for (int i = 0; i < result->height; i++) {
        for (int j = 0; j < result->width; j++) {
            fout << result->bytes[i * result->width + j] << " ";
        }
        fout << "\n";
    }

    std::cout << "Succes!\n";

    cleanup:
    if (fin.is_open()) {
        fin.close();
    }
    if (fout.is_open()) {
        fout.close();
    }
    destroyBitmap(image);
    destroyBitmap(kernel);
    destroyBitmap(result);

    return status;
}