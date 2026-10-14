/**
 * turboreg_cuda.cu
 *
 * CUDA kernels for GPU-accelerated cubic B-spline transformation and registration.
 * Port of TurboReg/StackReg algorithms to NVIDIA CUDA.
 *
 * Implements:
 * - 1D/2D/3D Recursive B-spline coefficient IIR filtering
 * - 2D and 3D batched cubic B-spline warping with mirror off-bounds boundaries
 *   supporting Translation, Rigid Body, Scaled Rotation, Affine, and Bilinear.
 */

extern "C" {

// B-spline pole z = sqrt(3) - 2
#define BSPLINE_Z -0.26794919243112270647255365849413
#define BSPLINE_LAMBDA 6.0

/**
 * 2D B-spline coefficient filter along rows.
 * Each thread handles one row y.
 */
__global__ void bspline_coeff_rows(
    double* __restrict__ img,
    int width,
    int height
) {
    int y = blockDim.x * blockIdx.x + threadIdx.x;
    if (y >= height) return;

    double* row = img + (long long)y * width;
    if (width <= 1) return;

    const double z = BSPLINE_Z;
    for (int n = 0; n < width; n++) {
        row[n] *= BSPLINE_LAMBDA;
    }

    double zn = pow(z, (double)width);
    double sum = (1.0 + z) * (row[0] + zn * row[width - 1]);
    double z1 = z;
    double zn2 = zn * zn;
    for (int n = 1; n < width - 1; n++) {
        z1 *= z;
        zn2 /= z;
        sum += (z1 + zn2) * row[n];
    }
    row[0] = sum / (1.0 - pow(z, (double)(2 * width)));

    // Causal pass
    for (int n = 1; n < width; n++) {
        row[n] += z * row[n - 1];
    }

    // Anti-causal pass
    row[width - 1] = z * row[width - 1] / (z - 1.0);
    for (int n = width - 2; n >= 0; n--) {
        row[n] = z * (row[n + 1] - row[n]);
    }
}

/**
 * 2D B-spline coefficient filter along columns.
 * Each thread handles one column x.
 */
__global__ void bspline_coeff_cols(
    double* __restrict__ img,
    int width,
    int height
) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x >= width) return;
    if (height <= 1) return;

    const double z = BSPLINE_Z;
    for (int n = 0; n < height; n++) {
        img[(long long)n * width + x] *= BSPLINE_LAMBDA;
    }

    double zn = pow(z, (double)height);
    double sum = (1.0 + z) * (img[x] + zn * img[(long long)(height - 1) * width + x]);
    double z1 = z;
    double zn2 = zn * zn;
    for (int n = 1; n < height - 1; n++) {
        z1 *= z;
        zn2 /= z;
        sum += (z1 + zn2) * img[(long long)n * width + x];
    }
    img[x] = sum / (1.0 - pow(z, (double)(2 * height)));

    // Causal pass
    for (int n = 1; n < height; n++) {
        img[(long long)n * width + x] += z * img[(long long)(n - 1) * width + x];
    }

    // Anti-causal pass
    img[(long long)(height - 1) * width + x] = z * img[(long long)(height - 1) * width + x] / (z - 1.0);
    for (int n = height - 2; n >= 0; n--) {
        img[(long long)n * width + x] = z * (img[(long long)(n + 1) * width + x] - img[(long long)n * width + x]);
    }
}

/**
 * 3D batched B-spline coefficient filter along rows across all frames.
 * Grid: (blockIdx.x: row blocks, blockIdx.y: frame index)
 */
__global__ void bspline_coeff_3d_rows(
    double* __restrict__ stack,
    int width,
    int height,
    int numFrames
) {
    int y = blockDim.x * blockIdx.x + threadIdx.x;
    int f = blockIdx.y;
    if (y >= height || f >= numFrames) return;

    double* row = stack + ((long long)f * height + y) * width;
    if (width <= 1) return;

    const double z = BSPLINE_Z;
    for (int n = 0; n < width; n++) {
        row[n] *= BSPLINE_LAMBDA;
    }

    double zn = pow(z, (double)width);
    double sum = (1.0 + z) * (row[0] + zn * row[width - 1]);
    double z1 = z;
    double zn2 = zn * zn;
    for (int n = 1; n < width - 1; n++) {
        z1 *= z;
        zn2 /= z;
        sum += (z1 + zn2) * row[n];
    }
    row[0] = sum / (1.0 - pow(z, (double)(2 * width)));

    // Causal pass
    for (int n = 1; n < width; n++) {
        row[n] += z * row[n - 1];
    }

    // Anti-causal pass
    row[width - 1] = z * row[width - 1] / (z - 1.0);
    for (int n = width - 2; n >= 0; n--) {
        row[n] = z * (row[n + 1] - row[n]);
    }
}

/**
 * 3D batched B-spline coefficient filter along columns across all frames.
 * Grid: (blockIdx.x: col blocks, blockIdx.y: frame index)
 */
__global__ void bspline_coeff_3d_cols(
    double* __restrict__ stack,
    int width,
    int height,
    int numFrames
) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int f = blockIdx.y;
    if (x >= width || f >= numFrames) return;
    if (height <= 1) return;

    double* frame = stack + (long long)f * height * width;
    const double z = BSPLINE_Z;
    for (int n = 0; n < height; n++) {
        frame[(long long)n * width + x] *= BSPLINE_LAMBDA;
    }

    double zn = pow(z, (double)height);
    double sum = (1.0 + z) * (frame[x] + zn * frame[(long long)(height - 1) * width + x]);
    double z1 = z;
    double zn2 = zn * zn;
    for (int n = 1; n < height - 1; n++) {
        z1 *= z;
        zn2 /= z;
        sum += (z1 + zn2) * frame[(long long)n * width + x];
    }
    frame[x] = sum / (1.0 - pow(z, (double)(2 * height)));

    // Causal pass
    for (int n = 1; n < height; n++) {
        frame[(long long)n * width + x] += z * frame[(long long)(n - 1) * width + x];
    }

    // Anti-causal pass
    frame[(long long)(height - 1) * width + x] = z * frame[(long long)(height - 1) * width + x] / (z - 1.0);
    for (int n = height - 2; n >= 0; n--) {
        frame[(long long)n * width + x] = z * (frame[(long long)(n + 1) * width + x] - frame[(long long)n * width + x]);
    }
}

/**
 * 2D Cubic B-spline transformation kernel with mirror off-bounds boundaries.
 * Parameters c0..c7 define:
 *   x = c0 + c1 * u + c2 * v + c3 * u * v
 *   y = c4 + c5 * u + c6 * v + c7 * u * v
 */
__global__ void bspline_transform_2d(
    const double* __restrict__ inCoeff,
    double* __restrict__ outImg,
    int inNx, int inNy,
    int outNx, int outNy,
    double c0, double c1, double c2, double c3,
    double c4, double c5, double c6, double c7
) {
    int u = blockDim.x * blockIdx.x + threadIdx.x;
    int v = blockDim.y * blockIdx.y + threadIdx.y;
    if (u >= outNx || v >= outNy) return;

    double x = c0 + c1 * u + c2 * v + c3 * u * v;
    double y = c4 + c5 * u + c6 * v + c7 * u * v;

    int xMsk = (0.0 <= x) ? ((int)(x + 0.5)) : ((int)(x - 0.5));
    int yMsk = (0.0 <= y) ? ((int)(y + 0.5)) : ((int)(y - 0.5));

    long long outIdx = (long long)v * outNx + u;
    if ((0 <= xMsk) && (xMsk < inNx) && (0 <= yMsk) && (yMsk < inNy)) {
        int twiceInNx = 2 * inNx;
        int twiceInNy = 2 * inNy;

        int px = (0.0 <= x) ? ((int)x + 2) : ((int)x + 1);
        int xIndex[4];
        for (int k = 0; k < 4; px--, k++) {
            int q = (px < 0) ? (-1 - px) : px;
            if (twiceInNx <= q) q -= twiceInNx * (q / twiceInNx);
            xIndex[k] = (inNx <= q) ? (twiceInNx - 1 - q) : q;
        }

        double rx = x - ((0.0 <= x) ? ((int)x) : ((int)x - 1));
        double sx = 1.0 - rx;
        double sxx = rx * rx;
        double xWeight[4];
        xWeight[3] = sx * sx * sx / 6.0;
        xWeight[2] = 2.0 / 3.0 - 0.5 * sxx * (2.0 - rx);
        xWeight[0] = sxx * rx / 6.0;
        xWeight[1] = 1.0 - xWeight[0] - xWeight[2] - xWeight[3];

        int py = (0.0 <= y) ? ((int)y + 2) : ((int)y + 1);
        int yIndex[4];
        for (int k = 0; k < 4; py--, k++) {
            int q = (py < 0) ? (-1 - py) : py;
            if (twiceInNy <= q) q -= twiceInNy * (q / twiceInNy);
            yIndex[k] = (inNy <= q) ? (twiceInNy - 1 - q) : q;
        }

        double ry = y - ((0.0 <= y) ? ((int)y) : ((int)y - 1));
        double sy = 1.0 - ry;
        double syy = ry * ry;
        double yWeight[4];
        yWeight[3] = sy * sy * sy / 6.0;
        yWeight[2] = 2.0 / 3.0 - 0.5 * syy * (2.0 - ry);
        yWeight[0] = syy * ry / 6.0;
        yWeight[1] = 1.0 - yWeight[0] - yWeight[2] - yWeight[3];

        double t = 0.0;
        for (int j = 0; j < 4; j++) {
            double s = 0.0;
            long long p = (long long)yIndex[j] * inNx;
            for (int i = 0; i < 4; i++) {
                s += xWeight[i] * inCoeff[p + xIndex[i]];
            }
            t += yWeight[j] * s;
        }
        outImg[outIdx] = (float)t;
    } else {
        outImg[outIdx] = 0.0;
    }
}

/**
 * 3D batched Cubic B-spline transformation kernel across all frames.
 * params has shape (numFrames, 8).
 */
__global__ void bspline_transform_3d(
    const double* __restrict__ inCoeffs,
    double* __restrict__ outImgs,
    const double* __restrict__ params,
    int inNx, int inNy,
    int outNx, int outNy,
    int numFrames
) {
    int u = blockDim.x * blockIdx.x + threadIdx.x;
    int v = blockDim.y * blockIdx.y + threadIdx.y;
    int f = blockIdx.z;

    if (u >= outNx || v >= outNy || f >= numFrames) return;

    const double* p = params + f * 8;
    double c0 = p[0], c1 = p[1], c2 = p[2], c3 = p[3];
    double c4 = p[4], c5 = p[5], c6 = p[6], c7 = p[7];

    double x = c0 + c1 * u + c2 * v + c3 * u * v;
    double y = c4 + c5 * u + c6 * v + c7 * u * v;

    int xMsk = (0.0 <= x) ? ((int)(x + 0.5)) : ((int)(x - 0.5));
    int yMsk = (0.0 <= y) ? ((int)(y + 0.5)) : ((int)(y - 0.5));

    long long frameOffset = (long long)f * inNx * inNy;
    long long outFrameOffset = (long long)f * outNx * outNy;
    long long outIdx = outFrameOffset + (long long)v * outNx + u;

    if ((0 <= xMsk) && (xMsk < inNx) && (0 <= yMsk) && (yMsk < inNy)) {
        int twiceInNx = 2 * inNx;
        int twiceInNy = 2 * inNy;

        int px = (0.0 <= x) ? ((int)x + 2) : ((int)x + 1);
        int xIndex[4];
        for (int k = 0; k < 4; px--, k++) {
            int q = (px < 0) ? (-1 - px) : px;
            if (twiceInNx <= q) q -= twiceInNx * (q / twiceInNx);
            xIndex[k] = (inNx <= q) ? (twiceInNx - 1 - q) : q;
        }

        double rx = x - ((0.0 <= x) ? ((int)x) : ((int)x - 1));
        double sx = 1.0 - rx;
        double sxx = rx * rx;
        double xWeight[4];
        xWeight[3] = sx * sx * sx / 6.0;
        xWeight[2] = 2.0 / 3.0 - 0.5 * sxx * (2.0 - rx);
        xWeight[0] = sxx * rx / 6.0;
        xWeight[1] = 1.0 - xWeight[0] - xWeight[2] - xWeight[3];

        int py = (0.0 <= y) ? ((int)y + 2) : ((int)y + 1);
        int yIndex[4];
        for (int k = 0; k < 4; py--, k++) {
            int q = (py < 0) ? (-1 - py) : py;
            if (twiceInNy <= q) q -= twiceInNy * (q / twiceInNy);
            yIndex[k] = (inNy <= q) ? (twiceInNy - 1 - q) : q;
        }

        double ry = y - ((0.0 <= y) ? ((int)y) : ((int)y - 1));
        double sy = 1.0 - ry;
        double syy = ry * ry;
        double yWeight[4];
        yWeight[3] = sy * sy * sy / 6.0;
        yWeight[2] = 2.0 / 3.0 - 0.5 * syy * (2.0 - ry);
        yWeight[0] = syy * ry / 6.0;
        yWeight[1] = 1.0 - yWeight[0] - yWeight[2] - yWeight[3];

        const double* inCoeff = inCoeffs + frameOffset;
        double t = 0.0;
        for (int j = 0; j < 4; j++) {
            double s = 0.0;
            long long rowIdx = (long long)yIndex[j] * inNx;
            for (int i = 0; i < 4; i++) {
                s += xWeight[i] * inCoeff[rowIdx + xIndex[i]];
            }
            t += yWeight[j] * s;
        }
        outImgs[outIdx] = (float)t;
    } else {
        outImgs[outIdx] = 0.0;
    }
}

} // extern "C"
