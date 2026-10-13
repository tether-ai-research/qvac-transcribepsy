// Fused depthwise conv2d (WHCN, F32), bit-exact with the CPU whcn order (ky-outer/kx-inner,
// OOB-skip). knl [KW,KH,1,C], data [W,H,C,N] -> dst [Wout,Hout,C,N]; threads over dst_x.
kernel void kernel_conv_2d_dw_whcn_f32(
    global const char * src, ulong off_src,
    global const char * knl, ulong off_knl,
    global       char * dst, ulong off_dst,
    int W, int H, int KW, int KH, int Wout, int Hout,
    int s0, int s1, int p0, int p1, int d0, int d1, int C
) {
    const int dx = get_global_id(0);
    if (dx >= Wout) return;
    const int dy = get_group_id(1);
    const int i  = get_group_id(2);     // c + C*b over channel*batch planes
    const int c  = i % C;               // one scalar mod per workgroup (not per-thread)

    global const float * sp = (global const float *)(src + off_src) + (long) i * W * H;
    global const float * kp = (global const float *)(knl + off_knl) + (long) c * KW * KH;
    global       float * dp = (global       float *)(dst + off_dst) + (long) i * Wout * Hout;

    float sum = 0.0f;
    for (int ky = 0; ky < KH; ++ky) {
        const int sy = dy * s1 + ky * d1 - p1;
        if (sy < 0 || sy >= H) continue;
        for (int kx = 0; kx < KW; ++kx) {
            const int sx = dx * s0 + kx * d0 - p0;
            if (sx < 0 || sx >= W) continue;
            sum += kp[ky * KW + kx] * sp[sy * W + sx];
        }
    }
    dp[dy * Wout + dx] = sum;
}
