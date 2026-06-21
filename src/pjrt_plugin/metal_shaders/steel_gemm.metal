// Steel GEMM kernel — offline-compiled version using MLX's Steel headers.
// This is compiled with `xcrun -sdk macosx metal` at build time for maximum
// compiler optimization (register allocation, instruction scheduling).
//
// The MLX Steel headers provide BlockLoader (vectorized cooperative loading)
// and BlockMMA (simdgroup_matrix multiply-accumulate with serpentine iteration).

#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#define STEEL_PRAGMA_NO_UNROLL _Pragma("clang loop unroll(disable)")

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

// ── Inline minimal Steel helpers (ported from MLX headers) ──

namespace steel {

template <typename T>
struct AccumHelper { typedef float accum_type; };

// Aligned ReadVector for coalesced device→threadgroup loads (from MLX loader.h)
template <typename T, short n_reads, short alignment = 1>
struct ReadVector {
    alignas(alignment * sizeof(T)) uint8_t v[sizeof(T) * n_reads];
};

// BlockLoader: cooperative tile loading from device to threadgroup memory
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    bool reduction_dim,
    short tgp_size,
    short n_reads = (BCOLS * BROWS) / tgp_size,
    short TCOLS = BCOLS / n_reads,
    short TROWS = tgp_size / TCOLS>
struct BlockLoader {
    static constant constexpr short vec_size = n_reads;

    const int src_ld;
    const int tile_stride;

    const short thread_idx;
    const short bi;
    const short bj;

    threadgroup T* dst;
    const device T* src;

    using RV = ReadVector<T, n_reads, n_reads>;

    BlockLoader(
        const device T* src_,
        const int src_ld_,
        threadgroup T* dst_,
        ushort simd_group_id,
        ushort simd_lane_id)
        : src_ld(src_ld_),
          tile_stride(reduction_dim ? BCOLS : BROWS * src_ld),
          thread_idx(simd_group_id * 32 + simd_lane_id),
          bi(thread_idx / TCOLS),
          bj(vec_size * (thread_idx % TCOLS)),
          dst(dst_ + bi * dst_ld + bj),
          src(src_ + bi * src_ld + bj) {}

    void load_unsafe() const {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < BROWS; i += TROWS) {
            *((threadgroup RV*)(&dst[i * dst_ld])) =
                *((const device RV*)(&src[i * src_ld]));
        }
    }

    void load_safe(short2 src_tile_dim) const {
        src_tile_dim = src_tile_dim - short2(bj, bi);
        if (src_tile_dim.x <= 0 || src_tile_dim.y <= 0) {
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < BROWS; i += TROWS) {
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < vec_size; j++) {
                    dst[i * dst_ld + j] = T(0);
                }
            }
            return;
        }
        bool tmp_idx[vec_size];
        T tmp_val[vec_size];
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < BROWS; i += TROWS) {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < vec_size; j++) {
                tmp_idx[j] = (i < src_tile_dim.y) && (j < src_tile_dim.x);
            }
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < vec_size; j++) {
                tmp_val[j] = src[(tmp_idx[j] ? i * src_ld + j : 0)];
            }
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < vec_size; j++) {
                dst[i * dst_ld + j] = tmp_idx[j] ? tmp_val[j] : T(0);
            }
        }
    }

    void next() { src += tile_stride; }
};

// BlockMMA: simdgroup_matrix multiply-accumulate (from MLX mma.h)
template <
    typename T,
    typename U,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    short lda_tgp,
    short ldb_tgp,
    typename AccumType = float>
struct BlockMMA {
    static constant constexpr short kFragSize = 8;

    static constant constexpr short TM_stride = kFragSize * WM;
    static constant constexpr short TN_stride = kFragSize * WN;
    static constant constexpr short TM = BM / (kFragSize * WM);
    static constant constexpr short TN = BN / (kFragSize * WN);

    static constant constexpr short A_str_m = lda_tgp;   // M
    static constant constexpr short A_str_k = 1;          // K
    static constant constexpr short B_str_k = ldb_tgp;    // K
    static constant constexpr short B_str_n = 1;          // N

    static constant constexpr short tile_stride_a = kFragSize * A_str_k;
    static constant constexpr short tile_stride_b = kFragSize * B_str_k;

    simdgroup_matrix<AccumType, 8, 8> Ctile[TM][TN];
    simdgroup_matrix<AccumType, 8, 8> Atile[TM];
    simdgroup_matrix<AccumType, 8, 8> Btile[TN];

    short sm;
    short sn;
    short As_offset;
    short Bs_offset;

    BlockMMA(
        ushort simd_group_id,
        ushort simd_lane_id) {
        short tm = kFragSize * (simd_group_id / WN);
        short tn = kFragSize * (simd_group_id % WN);

        // simdgroup_matrix thread coordinates
        const short qid = simd_lane_id / 4;
        const short fm = (qid & 4) + ((simd_lane_id / 2) % 4);
        const short fn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;

        sm = fm;
        sn = fn;

        As_offset = (tm + sm) * A_str_m + sn * A_str_k;
        Bs_offset = sm * B_str_k + (tn + sn) * B_str_n;

        sm += tm;
        sn += tn;

        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < TM; i++)
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < TN; j++)
                Ctile[i][j] = make_filled_simdgroup_matrix<AccumType, 8, 8>((AccumType)0);
    }

    void mma(const threadgroup T* As, const threadgroup T* Bs) {
        As += As_offset;
        Bs += Bs_offset;

        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < BK; kk += kFragSize) {
            simdgroup_barrier(mem_flags::mem_none);
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < TM; i++) {
                simdgroup_load(Atile[i], As + i * TM_stride * A_str_m, A_str_m);
            }
            simdgroup_barrier(mem_flags::mem_none);
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < TN; j++) {
                simdgroup_load(Btile[j], Bs + j * TN_stride * B_str_n, B_str_n);
            }
            simdgroup_barrier(mem_flags::mem_none);

            // Serpentine MMA
            STEEL_PRAGMA_UNROLL
            for (short m = 0; m < TM; m++) {
                STEEL_PRAGMA_UNROLL
                for (short n = 0; n < TN; n++) {
                    short n_serp = (m % 2) ? (TN - 1 - n) : n;
                    simdgroup_multiply_accumulate(Ctile[m][n_serp], Atile[m], Btile[n_serp], Ctile[m][n_serp]);
                }
            }

            As += tile_stride_a;
            Bs += tile_stride_b;
        }
    }

    void store_result(device U* D, const int ldd) {
        D += sm * ldd + sn;
        STEEL_PRAGMA_UNROLL
        for (short m = 0; m < TM; m++) {
            STEEL_PRAGMA_UNROLL
            for (short n = 0; n < TN; n++) {
                simdgroup_store(Ctile[m][n],
                    D + m * TM_stride * ldd + n * TN_stride, ldd);
            }
        }
    }
};

} // namespace steel

// ── GEMM kernel entry point ──
// Matches our existing buffer layout: A[M,K], B[K,N], C[M,N] (row-major)
// Uses MLX Steel BlockLoader + BlockMMA for maximum throughput.

template <
    typename T,
    typename U,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN>
[[kernel]] void mlx_gemm(
    device const T* A [[buffer(0)]],
    device const T* B [[buffer(1)]],
    device U* C       [[buffer(2)]],
    constant uint& M  [[buffer(3)]],
    constant uint& N_ [[buffer(4)]],
    constant uint& K  [[buffer(5)]],
    threadgroup T* As [[threadgroup(0)]],
    threadgroup T* Bs [[threadgroup(1)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {

    static constant constexpr short PAD = 16 / sizeof(T);
    static constant constexpr short A_LD = BK + PAD;
    static constant constexpr short B_LD = BN + PAD;
    static constant constexpr short TGP_SIZE = WM * WN * 32;

    const uint bm = tgid.x * BM;
    const uint bn = tgid.y * BN;
    const uint batch_z = tgid.z;

    using loader_a_t = steel::BlockLoader<T, BM, BK, A_LD, false, TGP_SIZE>;
    using loader_b_t = steel::BlockLoader<T, BK, BN, B_LD, true, TGP_SIZE>;
    using mma_t = steel::BlockMMA<T, U, BM, BN, BK, WM, WN, A_LD, B_LD>;

    device const T* A_ptr = A + batch_z * (size_t)M * K + (size_t)bm * K;
    device const T* B_ptr = B + batch_z * (size_t)K * N_ + bn;

    thread loader_a_t loader_a(A_ptr, K, As, simd_group_id, simd_lane_id);
    thread loader_b_t loader_b(B_ptr, N_, Bs, simd_group_id, simd_lane_id);
    thread mma_t mma_op(simd_group_id, simd_lane_id);

    int gemm_k_iterations = K / BK;

    for (int k = 0; k < gemm_k_iterations; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_a.load_unsafe();
        loader_b.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(As, Bs);
        loader_a.next();
        loader_b.next();
    }

    // Handle K remainder
    int k_rem = K - gemm_k_iterations * BK;
    if (k_rem > 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_a.load_safe(short2(k_rem, BM));
        loader_b.load_safe(short2(BN, k_rem));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(As, Bs);
    }

    mma_op.store_result(C + batch_z * (size_t)M * N_ + (size_t)bm * N_ + bn, N_);
}

// ── Template instantiations ──

// 64×64×16, WM=2, WN=2 (128 threads)
template [[host_name("mlx_gemm_64x64x16_f32")]] [[kernel]] void mlx_gemm<float, float, 64, 64, 16, 2, 2>(
    device const float*, device const float*, device float*,
    constant uint&, constant uint&, constant uint&,
    threadgroup float*, threadgroup float*, uint3, uint, uint);

template [[host_name("mlx_gemm_64x64x16_f16")]] [[kernel]] void mlx_gemm<half, half, 64, 64, 16, 2, 2>(
    device const half*, device const half*, device half*,
    constant uint&, constant uint&, constant uint&,
    threadgroup half*, threadgroup half*, uint3, uint, uint);

// 64×64×32, WM=2, WN=2 (128 threads)
template [[host_name("mlx_gemm_64x64x32_f32")]] [[kernel]] void mlx_gemm<float, float, 64, 64, 32, 2, 2>(
    device const float*, device const float*, device float*,
    constant uint&, constant uint&, constant uint&,
    threadgroup float*, threadgroup float*, uint3, uint, uint);

template [[host_name("mlx_gemm_64x64x32_f16")]] [[kernel]] void mlx_gemm<half, half, 64, 64, 32, 2, 2>(
    device const half*, device const half*, device half*,
    constant uint&, constant uint&, constant uint&,
    threadgroup half*, threadgroup half*, uint3, uint, uint);

// 128×64×32, WM=4, WN=2 (256 threads)
template [[host_name("mlx_gemm_128x64x32_f32")]] [[kernel]] void mlx_gemm<float, float, 128, 64, 32, 4, 2>(
    device const float*, device const float*, device float*,
    constant uint&, constant uint&, constant uint&,
    threadgroup float*, threadgroup float*, uint3, uint, uint);

template [[host_name("mlx_gemm_128x64x32_f16")]] [[kernel]] void mlx_gemm<half, half, 128, 64, 32, 4, 2>(
    device const half*, device const half*, device half*,
    constant uint&, constant uint&, constant uint&,
    threadgroup half*, threadgroup half*, uint3, uint, uint);

// 64×128×32, WM=2, WN=4 (256 threads)
template [[host_name("mlx_gemm_64x128x32_f32")]] [[kernel]] void mlx_gemm<float, float, 64, 128, 32, 2, 4>(
    device const float*, device const float*, device float*,
    constant uint&, constant uint&, constant uint&,
    threadgroup float*, threadgroup float*, uint3, uint, uint);

template [[host_name("mlx_gemm_64x128x32_f16")]] [[kernel]] void mlx_gemm<half, half, 64, 128, 32, 2, 4>(
    device const half*, device const half*, device half*,
    constant uint&, constant uint&, constant uint&,
    threadgroup half*, threadgroup half*, uint3, uint, uint);
