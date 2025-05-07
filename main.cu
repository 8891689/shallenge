/*
MIT License

Copyright (c) 2024 Nol Moonen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to furnish the Software to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Copyright (c) 2025 8891689

This code file contains modifications of the original work (Copyright (c) 2024 Nol Moonen).
Modifications include but are not limited to: manual expansion optimization of SHA-256 core functions, using CUDA constant memory to store constants and fixed messages, separation of SHA-256 implementations between host and device, and improvements in block-wise reduction logic.
*/

#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <cstdio> // 中文标注: 添加了cstdio头文件，用于fprintf

// 中文标注: 移除了原始码中的 DEVICE_UNROLL 宏定义，因为核心 SHA-256 循环将手动展开

// SHA-256 Constants (part of the algorithm)
// Declare device-side constants for arrays
// 中文标注: 将常数 K 分离为设备端 d_K (__constant__) 和主机端 h_K (constexpr)，并在main函数中拷贝
__constant__ uint32_t d_K[64];

// Initial hash values (part of the algorithm) - Use just constexpr for scalars
// 中文标注: 将初始哈希值 aa-hh 重命名为 H0-H7，仍为 constexpr
constexpr uint32_t H0 = 0x6a09e667;
constexpr uint32_t H1 = 0xbb67ae85;
constexpr uint32_t H2 = 0x3c6ef372;
constexpr uint32_t H3 = 0xa54ff53a;
constexpr uint32_t H4 = 0x510e527f;
constexpr uint32_t H5 = 0x9b05688c;
constexpr uint32_t H6 = 0x1f83d9ab;
constexpr uint32_t H7 = 0x5be0cd19;

// Host-side versions of K for copying (constexpr array)
// 中文标注: 这是主机端使用的 K 常数数组，用于拷贝到设备端
constexpr uint32_t h_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
    0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
    0x9bdc06a7, 0xc19bf174, 0xe49b69c1, /* REMOVED DUPLICATE in original */ 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
    0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
    0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
    0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2};


struct nonce_t {
    uint32_t m11;
    uint32_t m12;
    uint32_t m13;
};

struct hash_t {
    uint32_t arr[8];
};

// Basic SHA-256 bitwise functions as inline functions (for both host and device)
// 中文标注: 将位操作宏替换为 __forceinline__ __host__ __device__ 函数
__forceinline__ __host__ __device__ uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

__forceinline__ __host__ __device__ uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) | (z & (x ^ y));
}

__forceinline__ __host__ __device__ uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (z ^ (x & (y ^ z)));
}

__forceinline__ __host__ __device__ uint32_t ep0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

__forceinline__ __host__ __device__ uint32_t ep1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

__forceinline__ __host__ __device__ uint32_t sig0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

__forceinline__ __host__ __device__ uint32_t sig1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

// Optimized Endian Swap (same as original, essential for performance)
__forceinline__ __device__ __host__ uint32_t swap_endian(uint32_t x)
{
#ifdef __CUDA_ARCH__
    // Use efficient device intrinsic
    return __byte_perm(x, (uint32_t)0, (uint32_t)0x0123);
#else
    // Standard host implementation
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&x);
    return uint32_t{ptr[3]} | (uint32_t{ptr[2]} << 8) | (uint32_t{ptr[1]} << 16) |
           (uint32_t{ptr[0]} << 24);
#endif
}

// Fixed parts of the message block (same as original, defined by the problem)
// Declare device-side constants for arrays
// 中文标注: 将固定消息部分分离为设备端 d_fixed_msg (__constant__) 和主机端 h_fixed_msg (constexpr)，并在main函数中拷贝
__constant__ uint32_t d_fixed_msg[11];

// Host-side versions of fixed_msg for copying (constexpr array)
// 中文标注: 这是主机端使用的固定消息数组，用于拷贝到设备端
constexpr uint32_t h_fixed_msg[11] = {
    0x6e6f6c2f, // 'nol/'
    0x30303030, // '0000' * 10
    0x30303030,
    0x30303030,
    0x30303030,
    0x30303030,
    0x30303030,
    0x30303030,
    0x30303030,
    0x30303030,
    0x30303030
};

// Padding for SHA-256 (message length in bits) - Use just constexpr for scalars
// 中文标注: 填充常数 m14 和 m15 重命名为 padding_m14 和 padding_m15
constexpr uint32_t padding_m14 = 0x00000000; // upper part of u64 size
constexpr uint32_t padding_m15 = 0x000001b8; // length, 55 bytes = 440 bits (0x1b8)

// Helper macro for the core SHA-256 transformation step
// Using do-while(0) to create a single statement block for local variable scoping
// 中文标注: SHA-256 核心变换步骤宏
#define SHA256_TRANSFORM_STEP(a, b, c, d, e, f, g, h, w_i, k_i) do { \
    uint32_t T1 = h + ep1(e) + ch(e, f, g) + k_i + w_i;   \
    uint32_t T2 = ep0(a) + maj(a, b, c);                  \
    h = g; g = f; f = e; e = d + T1;                      \
    d = c; c = b; b = a; a = T1 + T2;                     \
} while(0)


// SHA-256 compression function implementation for DEVICE - Manual Unrolling
// 中文标注: 设备端 SHA-256 压缩函数实现 - 手动展开
__forceinline__ __device__ void sha256_manual_unroll_device(
    hash_t& hash, uint32_t m11, uint32_t m12, uint32_t m13)
{
    uint32_t a = H0, b = H1, c = H2, d = H3, e = H4, f = H5, g = H6, h = H7; // 中文标注: 使用重命名后的初始哈希值

    uint32_t W[16]; // Circular buffer for message schedule

    // Load initial 16 words into the buffer
    // 中文标注: 从设备端常量载入固定消息部分
    W[0] = d_fixed_msg[0]; W[1] = d_fixed_msg[1]; W[2] = d_fixed_msg[2]; W[3] = d_fixed_msg[3];
    W[4] = d_fixed_msg[4]; W[5] = d_fixed_msg[5]; W[6] = d_fixed_msg[6]; W[7] = d_fixed_msg[7];
    W[8] = d_fixed_msg[8]; W[9] = d_fixed_msg[9]; W[10] = d_fixed_msg[10];
    // 中文标注: 载入变量部分和填充部分
    W[11] = m11; W[12] = m12; W[13] = m13; W[14] = padding_m14; W[15] = padding_m15; // 中文标注: 使用重命名后的填充常数

    // --- Manual Unrolling of 64 Steps ---
    // 中文标注: 手动展开 64 步

    // Steps 0-15 (use initial W values)
    // 中文标注: 前 16 步使用初始 W 值
    SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[0], d_K[0]); // 中文标注: 使用设备端常数 d_K
    SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[1], d_K[1]);
    SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[2], d_K[2]);
    SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[3], d_K[3]);
    SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[4], d_K[4]);
    SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[5], d_K[5]);
    SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[6], d_K[6]);
    SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[7], d_K[7]);
    SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[8], d_K[8]);
    SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[9], d_K[9]);
    SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[10], d_K[10]);
    SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[11], d_K[11]);
    SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[12], d_K[12]);
    SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[13], d_K[13]);
    SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[14], d_K[14]);
    SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[15], d_K[15]);

    // Steps 16-63 (compute new W values and use them)
    // 中文标注: 步骤 16-63，计算新的 W 值并使用，使用循环缓冲区 W[0]-W[15]
    uint32_t w; // Temp variable for new W word

    // Step 16: i=16. W[16] = sig1(W[14]) + W[9] + sig0(W[1]) + W[0]. Store at W[0].
    w = sig1(W[14]) + W[9] + sig0(W[1]) + W[0]; W[0] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[0], d_K[16]);
    // Step 17: i=17. W[17] = sig1(W[15]) + W[10] + sig0(W[2]) + W[1]. Store at W[1].
    w = sig1(W[15]) + W[10] + sig0(W[2]) + W[1]; W[1] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[1], d_K[17]);
    // Step 18: i=18. W[18] = sig1(W[2]) + W[11] + sig0(W[3]) + W[2]. Store at W[2].
    w = sig1(W[2]) + W[11] + sig0(W[3]) + W[2]; W[2] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[2], d_K[18]);
    // Step 19: i=19. W[19] = sig1(W[3]) + W[12] + sig0(W[4]) + W[3]. Store at W[3].
    w = sig1(W[3]) + W[12] + sig0(W[4]) + W[3]; W[3] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[3], d_K[19]);
    // Step 20: i=20. W[20] = sig1(W[4]) + W[13] + sig0(W[5]) + W[4]. Store at W[4].
    w = sig1(W[4]) + W[13] + sig0(W[5]) + W[4]; W[4] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[4], d_K[20]);
    // Step 21: i=21. W[21] = sig1(W[5]) + W[14] + sig0(W[6]) + W[5]. Store at W[5].
    w = sig1(W[5]) + W[14] + sig0(W[6]) + W[5]; W[5] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[5], d_K[21]);
    // Step 22: i=22. W[22] = sig1(W[6]) + W[15] + sig0(W[7]) + W[6]. Store at W[6].
    w = sig1(W[6]) + W[15] + sig0(W[7]) + W[6]; W[6] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[6], d_K[22]);
    // Step 23: i=23. W[23] = sig1(W[7]) + W[0] + sig0(W[8]) + W[7]. Store at W[7]. (Indices wrap in W)
    w = sig1(W[7]) + W[0] + sig0(W[8]) + W[7]; W[7] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[7], d_K[23]);
    // Step 24: i=24. W[24] = sig1(W[8]) + W[1] + sig0(W[9]) + W[8]. Store at W[8].
    w = sig1(W[8]) + W[1] + sig0(W[9]) + W[8]; W[8] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[8], d_K[24]);
    // Step 25: i=25. W[25] = sig1(W[9]) + W[2] + sig0(W[10]) + W[9]. Store at W[9].
    w = sig1(W[9]) + W[2] + sig0(W[10]) + W[9]; W[9] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[9], d_K[25]);
    // Step 26: i=26. W[26] = sig1(W[10]) + W[3] + sig0(W[11]) + W[10]. Store at W[10].
    w = sig1(W[10]) + W[3] + sig0(W[11]) + W[10]; W[10] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[10], d_K[26]);
    // Step 27: i=27. W[27] = sig1(W[11]) + W[4] + sig0(W[12]) + W[11]. Store at W[11].
    w = sig1(W[11]) + W[4] + sig0(W[12]) + W[11]; W[11] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[11], d_K[27]);
    // Step 28: i=28. W[28] = sig1(W[12]) + W[5] + sig0(W[13]) + W[12]. Store at W[12].
    w = sig1(W[12]) + W[5] + sig0(W[13]) + W[12]; W[12] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[12], d_K[28]);
    // Step 29: i=29. W[29] = sig1(W[13]) + W[6] + sig0(W[14]) + W[13]. Store at W[13].
    w = sig1(W[13]) + W[6] + sig0(W[14]) + W[13]; W[13] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[13], d_K[29]);
    // Step 30: i=30. W[30] = sig1(W[14]) + W[7] + sig0(W[15]) + W[14]. Store at W[14].
    w = sig1(W[14]) + W[7] + sig0(W[15]) + W[14]; W[14] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[14], d_K[30]);
    // Step 31: i=31. W[31] = sig1(W[15]) + W[8] + sig0(W[0]) + W[15]. Store at W[15].
    w = sig1(W[15]) + W[8] + sig0(W[0]) + W[15]; W[15] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[15], d_K[31]);

    // Steps 32-47
    w = sig1(W[0]) + W[9] + sig0(W[1]) + W[0]; W[0] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[0], d_K[32]);
    w = sig1(W[1]) + W[10] + sig0(W[2]) + W[1]; W[1] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[1], d_K[33]);
    w = sig1(W[2]) + W[11] + sig0(W[3]) + W[2]; W[2] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[2], d_K[34]);
    w = sig1(W[3]) + W[12] + sig0(W[4]) + W[3]; W[3] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[3], d_K[35]);
    w = sig1(W[4]) + W[13] + sig0(W[5]) + W[4]; W[4] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[4], d_K[36]);
    w = sig1(W[5]) + W[14] + sig0(W[6]) + W[5]; W[5] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[5], d_K[37]);
    w = sig1(W[6]) + W[15] + sig0(W[7]) + W[6]; W[6] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[6], d_K[38]);
    w = sig1(W[7]) + W[0] + sig0(W[8]) + W[7]; W[7] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[7], d_K[39]);
    w = sig1(W[8]) + W[1] + sig0(W[9]) + W[8]; W[8] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[8], d_K[40]);
    w = sig1(W[9]) + W[2] + sig0(W[10]) + W[9]; W[9] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[9], d_K[41]);
    w = sig1(W[10]) + W[3] + sig0(W[11]) + W[10]; W[10] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[10], d_K[42]);
    w = sig1(W[11]) + W[4] + sig0(W[12]) + W[11]; W[11] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[11], d_K[43]);
    w = sig1(W[12]) + W[5] + sig0(W[13]) + W[12]; W[12] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[12], d_K[44]);
    w = sig1(W[13]) + W[6] + sig0(W[14]) + W[13]; W[13] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[13], d_K[45]);
    w = sig1(W[14]) + W[7] + sig0(W[15]) + W[14]; W[14] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[14], d_K[46]);
    w = sig1(W[15]) + W[8] + sig0(W[0]) + W[15]; W[15] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[15], d_K[47]);

    // Steps 48-63
    w = sig1(W[0]) + W[9] + sig0(W[1]) + W[0]; W[0] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[0], d_K[48]);
    w = sig1(W[1]) + W[10] + sig0(W[2]) + W[1]; W[1] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[1], d_K[49]);
    w = sig1(W[2]) + W[11] + sig0(W[3]) + W[2]; W[2] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[2], d_K[50]);
    w = sig1(W[3]) + W[12] + sig0(W[4]) + W[3]; W[3] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[3], d_K[51]);
    w = sig1(W[4]) + W[13] + sig0(W[5]) + W[4]; W[4] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[4], d_K[52]);
    w = sig1(W[5]) + W[14] + sig0(W[6]) + W[5]; W[5] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[5], d_K[53]);
    w = sig1(W[6]) + W[15] + sig0(W[7]) + W[6]; W[6] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[6], d_K[54]);
    w = sig1(W[7]) + W[0] + sig0(W[8]) + W[7]; W[7] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[7], d_K[55]);
    w = sig1(W[8]) + W[1] + sig0(W[9]) + W[8]; W[8] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[8], d_K[56]);
    w = sig1(W[9]) + W[2] + sig0(W[10]) + W[9]; W[9] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[9], d_K[57]);
    w = sig1(W[10]) + W[3] + sig0(W[11]) + W[10]; W[10] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[10], d_K[58]);
    w = sig1(W[11]) + W[4] + sig0(W[12]) + W[11]; W[11] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[11], d_K[59]);
    w = sig1(W[12]) + W[5] + sig0(W[13]) + W[12]; W[12] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[12], d_K[60]);
    w = sig1(W[13]) + W[6] + sig0(W[14]) + W[13]; W[13] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[13], d_K[61]);
    w = sig1(W[14]) + W[7] + sig0(W[15]) + W[14]; W[14] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[14], d_K[62]);
    w = sig1(W[15]) + W[8] + sig0(W[0]) + W[15]; W[15] = w; SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[15], d_K[63]);


    // --- End of Manual Unrolling ---


    // Add the final hash values to the initial values
    // 中文标注: 将最终哈希值加到初始哈希值上
    hash.arr[0] = H0 + a; // 中文标注: 使用重命名后的初始哈希值
    hash.arr[1] = H1 + b;
    hash.arr[2] = H2 + c;
    hash.arr[3] = H3 + d;
    hash.arr[4] = H4 + e;
    hash.arr[5] = H5 + f;
    hash.arr[6] = H6 + g;
    hash.arr[7] = H7 + h;
}

// SHA-256 compression function implementation for HOST (Standard loop version)
// 中文标注: 主机端 SHA-256 压缩函数实现 (标准循环版本)
__forceinline__ __host__ void sha256_host(
    hash_t& hash, uint32_t m11, uint32_t m12, uint32_t m13)
{
    uint32_t a = H0; // 中文标注: 使用重命名后的初始哈希值
    uint32_t b = H1;
    uint32_t c = H2;
    uint32_t d = H3;
    uint32_t e = H4;
    uint32_t f = H5;
    uint32_t g = H6;
    uint32_t h = H7;

    uint32_t W[64]; // Host version can use the full W array easily

    // Load initial 16 words (from input message)
    for (int i = 0; i < 11; ++i) {
        W[i] = h_fixed_msg[i]; // 中文标注: 使用主机端常量 h_fixed_msg
    }
    W[11] = m11;
    W[12] = m12;
    W[13] = m13;
    W[14] = padding_m14; // 中文标注: 使用重命名后的填充常数
    W[15] = padding_m15;

    // Compute remaining 48 words
    for (int i = 16; i < 64; ++i) {
        W[i] = sig1(W[i - 2]) + W[i - 7] + sig0(W[i - 15]) + W[i - 16]; // 中文标注: 使用新的位操作函数
    }


    // SHA-256 Compression loop (64 steps)
    for (int i = 0; i < 64; ++i) {
        // Perform the core SHA-256 step using the macro (it's safe inside a loop too)
        SHA256_TRANSFORM_STEP(a,b,c,d,e,f,g,h, W[i], h_K[i]); // 中文标注: 使用主机端常数 h_K 和新的变换宏
    }

    // Add the final hash values to the initial values
    // 中文标注: 将最终哈希值加到初始哈希值上
    hash.arr[0] = H0 + a; // 中文标注: 使用重命名后的初始哈希值
    hash.arr[1] = H1 + b;
    hash.arr[2] = H2 + c;
    hash.arr[3] = H3 + d;
    hash.arr[4] = H4 + e;
    hash.arr[5] = H5 + f;
    hash.arr[6] = H6 + g;
    hash.arr[7] = H7 + h;
}


// Comparison, Copy, Set Worst Value (logic is the same)
// These need to be __host__ __device__ as they are used in both kernel and main
__forceinline__ __host__ __device__ bool less_than(const hash_t& lhs, const hash_t& rhs)
{
    // No PRAGMA_UNROLL needed here, it's a small loop and compiler can handle it.
    // 中文标注: 移除了原始码中关于 DEVICE_UNROLL 宏的注释
    for (int i = 0; i < 8; ++i) {
        if (lhs.arr[i] < rhs.arr[i]) {
            return true;
        } else if (rhs.arr[i] < lhs.arr[i]) {
            return false;
        }
    }
    return false;
}

__forceinline__ __host__ __device__ void copy(hash_t& dst, const hash_t& src)
{
    // Prefer memcpy if available and efficient, otherwise manual copy
#ifdef __CUDA_ARCH__
    // On device, memcpy is generally efficient
    std::memcpy(&dst, &src, sizeof(hash_t));
#else
    // On host, memcpy is standard
    std::memcpy(&dst, &src, sizeof(hash_t));
#endif
}

__forceinline__ __host__ __device__ void set_worst_hash_value(hash_t& hash)
{
    // Prefer memset if available and efficient
#ifdef __CUDA_ARCH__
    // On device, memset might be less efficient than a loop depending on context
    // A loop might be better for registers
    // No PRAGMA_UNROLL needed here either.
    for(int i=0; i<8; ++i) hash.arr[i] = 0xffffffff; // 中文标注: 使用循环设置最差哈希值，对于设备端可能更好
#else
    // On host, memset is standard and efficient
    std::memset(&hash, 0xff, sizeof(hash_t)); // 中文标注: 主机端使用 memset
#endif
}

// Base64 encoding logic (same as original, tied to the problem)
// Marked __device__ as it's primarily used in the kernel
// 中文标注: base64 编码逻辑
__device__ constexpr int base64_max = 62; // a-z, A-Z, 0-9

// 中文标注: base64 到 ascii 转换函数，现在仅标记为 __device__
__forceinline__ __device__ uint8_t base64_to_ascii(int x)
{
    assert(0 <= x && x < base64_max); // 中文标注: 添加断言
    // Using built-in hint for better performance
    __builtin_assume(0 <= x && x < 62); // 中文标注: 添加编译器提示
    if (x < 26) return 'A' + x;
    if (x < 52) return 'a' + (x - 26);
    return '0' + (x - 52); // 中文标注: base64 映射逻辑与原始码略有不同，但效果相同
}

__device__ constexpr int max_thread_count = base64_max * base64_max * base64_max * base64_max;

/// \brief Encode a value in range [0, base64_max^4) to a u32 encoded as base64.
// 中文标注: encode 函数，现在仅标记为 __device__
__forceinline__ __device__ uint32_t encode(int val)
{
    assert(0 <= val && val < max_thread_count);
    uint32_t ret{};
    for (int i = 0; i < 4; ++i) {
        ret |= (uint32_t)base64_to_ascii(val % base64_max) << (i * 8); // Note the byte order for u32
        val /= base64_max;
    }
     return ret;
}

// 中文标注: 内核函数，名称从 hash 更改为 hash_manual_unroll，并增加了 __launch_bounds__
template <int block_size>
__global__ void __launch_bounds__(block_size) hash_manual_unroll(int iteration, nonce_t* nonces)
{
    // set m11 (iteration) and m12 (thread index)
    const uint32_t m11 = encode(iteration);
    const int idx      = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t m12 = encode(idx);

    hash_t thread_best_hash{};
    set_worst_hash_value(thread_best_hash);
    uint32_t thread_best_m13{};

    // Iterate over possible values for the first 3 bytes of m13
    // These loops are small enough that manual unrolling isn't strictly needed,
    // but can be added with PRAGMA_UNROLL if desired for tiny gains.
    // PRAGMA_UNROLL // Could add here
    for (int i = 0; i < base64_max; ++i) {
    // PRAGMA_UNROLL // Could add here
        for (int j = 0; j < base64_max; ++j) {
        // PRAGMA_UNROLL // Could add here
            for (int k = 0; k < base64_max; ++k) {
                // Construct the variable part of m13, including the required 0x80 padding bit
                const uint32_t m13 = ((uint32_t)base64_to_ascii(i) << 24) |
                                     ((uint32_t)base64_to_ascii(j) << 16) |
                                     ((uint32_t)base64_to_ascii(k) << 8) |
                                     uint32_t{0x80}; // Padding bit

                hash_t current_hash;
                // Call the manually unrolled device version
                // 中文标注: 调用手动展开的设备端哈希函数
                sha256_manual_unroll_device(current_hash, m11, m12, m13);

                if (less_than(current_hash, thread_best_hash)) {
                    copy(thread_best_hash, current_hash);
                    thread_best_m13 = m13;
                }
            }
        }
    }

    // Reduce results within the block to find the best hash and corresponding m13
    // 中文标注: 块内归约，寻找块内最优哈希值和对应的 m13
    struct reduction_type_v2 { // 中文标注: 新的归约结构体，直接携带 m13
        hash_t hash;
        uint32_t m13; // Carry the m13 value along with the hash
    };

    reduction_type_v2 val_v2;
    copy(val_v2.hash, thread_best_hash);
    val_v2.m13 = thread_best_m13; // Store the best m13 found by *this* thread

    using block_reduce_v2 = cub::BlockReduce<reduction_type_v2, block_size>;
    __shared__ typename block_reduce_v2::TempStorage tmp_storage_v2;
    // Use the __device__ lambda with __attribute__((device))
    // 中文标注: 使用设备端 lambda 函数进行归约，并添加属性
    const reduction_type_v2 block_best_v2 =
        block_reduce_v2(tmp_storage_v2)
            .Reduce(
                val_v2,
                [] __attribute__((device))(const reduction_type_v2& lhs, const reduction_type_v2& rhs)
                    -> reduction_type_v2 { return less_than(lhs.hash, rhs.hash) ? lhs : rhs; });


    // The first thread in the block stores the block's best result
    if (threadIdx.x == 0) {
        nonces[blockIdx.x].m11 = m11;
        nonces[blockIdx.x].m12 = m12;
        nonces[blockIdx.x].m13 = block_best_v2.m13; // 中文标注: 存储归约结果中携带的最优 m13
    }
}


// Host side helper for error checking (different macro name)
// 中文标注: 主机端错误检查辅助宏，名称从 CHECK_CUDA 改为 CUDA_CHECK_ERR
#define CUDA_CHECK_ERR(call)                                                                           \
    do {                                                                                           \
        cudaError_t err = call;                                                                    \
        if (err != cudaSuccess) {                                                                  \
            fprintf(stderr, "CUDA error at %s:%d \"%s\"\n", __FILE__, __LINE__, cudaGetErrorString(err));   \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (0)

// Host side printing functions (slightly different implementation)
// 中文标注: 主机端打印函数，名称末尾添加了 _alt
void print_u32_as_char_alt(uint32_t x, int n = 4)
{
    uint32_t tmp = swap_endian(x); // Swap for printing characters in order
    char buffer[5];
    std::memcpy(buffer, &tmp, 4);
    buffer[4] = '\0';
    for (int j = 0; j < n; ++j) {
        fputc(buffer[j], stdout); // 中文标注: 使用 fputc 替代 printf 打印字符
    }
}

// 中文标注: 主机端打印输入函数，使用 h_fixed_msg 数组
void print_input_alt(const nonce_t& nonce)
{
    for(int i=0; i<11; ++i) print_u32_as_char_alt(h_fixed_msg[i]); // 中文标注: 使用主机端常量 h_fixed_msg 打印固定消息
    print_u32_as_char_alt(nonce.m11);
    print_u32_as_char_alt(nonce.m12);
    print_u32_as_char_alt(nonce.m13, 3); // Only print the first 3 bytes of m13
    printf("\n");
}

// 中文标注: 主机端打印哈希函数
void print_hash_alt(const hash_t& hash)
{
    for (int i = 0; i < 8; ++i) {
        printf("%08x ", hash.arr[i]); // 中文标注: 使用 %08x 格式打印
    }
    printf("\n");
}

int main(int argc, char* argv[])
{
    setbuf(stdout, nullptr); // make stream unbuffered

    int iter_offset = 0;
    if (argc > 1) {
        iter_offset = std::strtol(argv[1], nullptr, 10);
    }

    // --- Copy constants to device memory ---
    // 中文标注: 将主机端常数拷贝到设备端
    CUDA_CHECK_ERR(cudaMemcpyToSymbol(d_K, h_K, sizeof(h_K)));
    CUDA_CHECK_ERR(cudaMemcpyToSymbol(d_fixed_msg, h_fixed_msg, sizeof(h_fixed_msg)));
    // ---------------------------------------

    cudaStream_t stream;
    CUDA_CHECK_ERR(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    CUDA_CHECK_ERR(cudaEventCreate(&start));
    CUDA_CHECK_ERR(cudaEventCreate(&stop));

    constexpr int grid_size  = 256;
    constexpr int block_size = 256;
    // Use host version for static assert
    // 中文标注: 静态断言使用主机端常量 base64_max
    static_assert(grid_size * block_size <= 62 * 62 * 62 * 62); // Use constant from host for calculation


    nonce_t* d_nonces{};
    CUDA_CHECK_ERR(cudaMalloc(&d_nonces, grid_size * sizeof(nonce_t)));

    nonce_t best_nonce{};
    hash_t overall_best_hash; // 中文标注: 变量名从 best_hash 改为 overall_best_hash
    set_worst_hash_value(overall_best_hash);

    const int num_batches_to_run = INT_MAX; // Run until interrupted // 中文标注: 变量名从 num_batches 改为 num_batches_to_run
    const int iters_per_batch = 2; // Number of iterations per timing measurement // 中文标注: 变量名从 num_iters_per_batch 改为 iters_per_batch

    for (int batch_idx = 0; batch_idx < num_batches_to_run; ++batch_idx) {

        CUDA_CHECK_ERR(cudaEventRecord(start, stream));

        // Launch kernel for each iteration in the batch
        for (int j = 0; j < iters_per_batch; ++j) {
            const int current_iteration = iter_offset + iters_per_batch * batch_idx + j;
            // Call the kernel using the manual unroll version
            // 中文标注: 调用新命名的内核函数 hash_manual_unroll，并传递 stream
            hash_manual_unroll<block_size><<<grid_size, block_size, 0 /* shared memory */, stream>>>(
                current_iteration, d_nonces);
            CUDA_CHECK_ERR(cudaGetLastError()); // 中文标注: 在内核启动后立即检查错误
        }

        CUDA_CHECK_ERR(cudaEventRecord(stop, stream));

        // Wait for the batch to complete and measure time
        CUDA_CHECK_ERR(cudaEventSynchronize(stop));
        float milliseconds{}; // 中文标注: 变量名从 milliseconds 改为 milliseconds
        CUDA_CHECK_ERR(cudaEventElapsedTime(&milliseconds, start, stop));

        // Calculate hash rate
        // 中文标注: 哈希率计算使用主机端常量 base64_max
        const double hashes_calculated_per_batch = static_cast<double>(iters_per_batch) * grid_size *
                                                 block_size * 62 * 62 * 62; // Use constant from host for calculation
        const double seconds = milliseconds / 1000.0;
        printf(
            "iter [%d, %d): %fGH/s (%fms)\n",
            iter_offset + iters_per_batch * batch_idx,
            iter_offset + iters_per_batch * (batch_idx + 1),
            hashes_calculated_per_batch / seconds / 1.e9, // Convert to GH/s
            milliseconds);

        // Copy block best nonces from device to host
        std::vector<nonce_t> h_block_nonces(grid_size); // 中文标注: 变量名从 h_nonces 改为 h_block_nonces
        CUDA_CHECK_ERR(cudaMemcpy(
            h_block_nonces.data(), d_nonces, grid_size * sizeof(nonce_t), cudaMemcpyDeviceToHost));

        // Check block best nonces and update overall best
        for (int i = 0; i < grid_size; ++i) {
            hash_t current_block_hash{};
            // Calculate hash on host using the nonce found by the block
            // Call the host version of the hash function
            // 中文标注: 主机端验证改用 sha256_host 函数
            sha256_host(current_block_hash, h_block_nonces[i].m11, h_block_nonces[i].m12, h_block_nonces[i].m13);

            if (less_than(current_block_hash, overall_best_hash)) { // 中文标注: 变量名整体更新
                best_nonce = h_block_nonces[i];
                copy(overall_best_hash, current_block_hash); // 中文标注: 变量名整体更新
                printf("Found new best:\n");
                print_input_alt(best_nonce); // 中文标注: 调用新命名的打印函数
                print_hash_alt(overall_best_hash); // 中文标注: 调用新命名的打印函数
            }
        }
    }

    printf("final result:\n");
    print_input_alt(best_nonce); // 中文标注: 调用新命名的打印函数
    print_hash_alt(overall_best_hash); // 中文标注: 调用新命名的打印函数

    CUDA_CHECK_ERR(cudaFree(d_nonces)); // 中文标注: 变量名更新
    CUDA_CHECK_ERR(cudaEventDestroy(stop)); // 中文标注: 变量名更新
    CUDA_CHECK_ERR(cudaEventDestroy(start)); // 中文标注: 变量名更新
    CUDA_CHECK_ERR(cudaStreamDestroy(stream)); // 中文标注: 变量名更新

    return 0;
}
