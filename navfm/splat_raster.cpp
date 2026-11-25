#include <algorithm>
#include <cmath>
#include <vector>

namespace {
struct Splat {
    float x, y, z, a, b, c, rx, ry;
    const float* rgb;
};
}

extern "C" void splat_raster(const float* cloud, int count, const float* view,
                            const float* intrinsics, int w, int h, float* pixels) {
    const float fx = intrinsics[0], fy = intrinsics[1], cx = intrinsics[2], cy = intrinsics[3];
    std::vector<Splat> splats;
    splats.reserve(count);
    auto rotate = [&](int row, float x, float y, float z) {
        return view[3 + 3 * row] * x + view[4 + 3 * row] * y + view[5 + 3 * row] * z;
    };
    for (int i = 0; i < count; ++i) {
        const float* p = cloud + 11 * i;
        float dx = p[0] - view[0], dy = p[1] - view[1], dz = p[2] - view[2];
        if (dx * p[3] + dy * p[4] + dz * p[5] >= 0) continue;
        float z = rotate(2, dx, dy, dz);
        if (z < .055f) continue;
        float x = rotate(0, dx, dy, dz) / z, y = rotate(1, dx, dy, dz) / z;
        float u = cx + fx * x, v = cy + fy * y;
        if (u < -48 || v < -48 || u > w + 48 || v > h + 48) continue;
        // Project the surface-aligned covariance using the perspective Jacobian.
        float nz = rotate(2, p[3], p[4], p[5]);
        float jx = rotate(0, p[3], p[4], p[5]) - x * nz;
        float jy = rotate(1, p[3], p[4], p[5]) - y * nz;
        float t = p[9] * p[9], d = p[10] * p[10] - t;
        float kx = fx / z, ky = fy / z;
        float xx = .24f + kx * kx * (t * (1 + x * x) + d * jx * jx);
        float yy = .24f + ky * ky * (t * (1 + y * y) + d * jy * jy);
        float xy = kx * ky * (t * x * y + d * jx * jy);
        float det = xx * yy - xy * xy;
        splats.push_back({u, v, z, yy / det, -xy / det, xx / det,
                          std::min(48.f, 3 * std::sqrt(xx)), std::min(48.f, 3 * std::sqrt(yy)), p + 6});
    }
    std::sort(splats.begin(), splats.end(), [](const Splat& a, const Splat& b) { return a.z < b.z; });
    std::vector<float> transmittance(w * h, 1);
    for (const auto& s : splats) {
        int left = std::max(0, int(std::floor(s.x - s.rx)));
        int right = std::min(w - 1, int(std::ceil(s.x + s.rx)));
        int top = std::max(0, int(std::floor(s.y - s.ry)));
        int bottom = std::min(h - 1, int(std::ceil(s.y + s.ry)));
        for (int y = top; y <= bottom; ++y) {
            float dy = y + .5f - s.y;
            for (int x = left; x <= right; ++x) {
                int i = y * w + x;
                if (transmittance[i] < .005f) continue;
                float dx = x + .5f - s.x;
                float q = s.a * dx * dx + 2 * s.b * dx * dy + s.c * dy * dy;
                if (q > 9) continue;
                float opacity = .92f * std::exp(-.5f * q);
                float weight = transmittance[i] * opacity;
                for (int k = 0; k < 3; ++k) pixels[i * 3 + k] += weight * s.rgb[k];
                transmittance[i] *= 1 - opacity;
            }
        }
    }
    const float background[] = {.21f, .24f, .28f};
    for (int i = 0; i < w * h; ++i) {
        for (int k = 0; k < 3; ++k) pixels[i * 3 + k] += transmittance[i] * background[k];
    }
}
