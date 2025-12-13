#pragma once
#include <cstdint>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace minisolver {

// SIMD capability levels
enum class SimdLevel {
    NONE = 0,
    AVX2 = 1,
    AVX512 = 2
};

// CPU capability detection
class SimdDetector {
public:
    static SimdLevel detect_capability() {
        static SimdLevel cached_level = detect_capability_impl();
        return cached_level;
    }

private:
    static SimdLevel detect_capability_impl() {
        // Check AVX512 first (more advanced)
#ifdef __AVX512F__
        // Runtime check for AVX512 support
        if (__builtin_cpu_supports("avx512f")) {
            return SimdLevel::AVX512;
        }
#endif

        // Check AVX2
#ifdef __AVX2__
        if (__builtin_cpu_supports("avx2")) {
            return SimdLevel::AVX2;
        }
#endif

        return SimdLevel::NONE;
    }
};

// SIMD math operations
template<typename T, SimdLevel Level>
class SimdOps {
public:
    static constexpr int vector_width = 1; // Scalar fallback
    using vector_type = T;

    static vector_type load(const T* ptr) { return *ptr; }
    static void store(T* ptr, vector_type val) { *ptr = val; }

    static vector_type add(vector_type a, vector_type b) { return a + b; }
    static vector_type sub(vector_type a, vector_type b) { return a - b; }
    static vector_type mul(vector_type a, vector_type b) { return a * b; }
    static vector_type abs(vector_type a) { return std::abs(a); }
    static vector_type log(vector_type a) { return std::log(a); }
    static vector_type max(vector_type a, vector_type b) { return std::max(a, b); }
    static vector_type min(vector_type a, vector_type b) { return std::min(a, b); }
};

// AVX2 specialization
#ifdef __AVX2__
template<>
class SimdOps<double, SimdLevel::AVX2> {
public:
    static constexpr int vector_width = 4; // AVX2 can handle 4 doubles
    using vector_type = __m256d;

    static vector_type load(const double* ptr) { return _mm256_loadu_pd(ptr); }
    static void store(double* ptr, vector_type val) { _mm256_storeu_pd(ptr, val); }

    static vector_type add(vector_type a, vector_type b) { return _mm256_add_pd(a, b); }
    static vector_type sub(vector_type a, vector_type b) { return _mm256_sub_pd(a, b); }
    static vector_type mul(vector_type a, vector_type b) { return _mm256_mul_pd(a, b); }
    static vector_type abs(vector_type a) {
        __m256d mask = _mm256_set1_pd(-0.0); // Negative zero for sign bit mask
        return _mm256_andnot_pd(mask, a);
    }
    static vector_type log(vector_type a) {
        // Fallback to scalar for transcendental functions
        double temp[4];
        _mm256_storeu_pd(temp, a);
        temp[0] = std::log(temp[0]);
        temp[1] = std::log(temp[1]);
        temp[2] = std::log(temp[2]);
        temp[3] = std::log(temp[3]);
        return _mm256_loadu_pd(temp);
    }
    static vector_type max(vector_type a, vector_type b) { return _mm256_max_pd(a, b); }
    static vector_type min(vector_type a, vector_type b) { return _mm256_min_pd(a, b); }
    static vector_type set1(double val) { return _mm256_set1_pd(val); }
};
#endif

// AVX512 specialization
#ifdef __AVX512F__
template<>
class SimdOps<double, SimdLevel::AVX512> {
public:
    static constexpr int vector_width = 8; // AVX512 can handle 8 doubles
    using vector_type = __m512d;

    static vector_type load(const double* ptr) { return _mm512_loadu_pd(ptr); }
    static void store(double* ptr, vector_type val) { _mm512_storeu_pd(ptr, val); }

    static vector_type add(vector_type a, vector_type b) { return _mm512_add_pd(a, b); }
    static vector_type sub(vector_type a, vector_type b) { return _mm512_sub_pd(a, b); }
    static vector_type mul(vector_type a, vector_type b) { return _mm512_mul_pd(a, b); }
    static vector_type abs(vector_type a) {
        __m512d mask = _mm512_set1_pd(-0.0); // Negative zero for sign bit mask
        return _mm512_andnot_pd(mask, a);
    }
    static vector_type log(vector_type a) {
        // Fallback to scalar for transcendental functions
        double temp[8];
        _mm512_storeu_pd(temp, a);
        for(int i = 0; i < 8; ++i) temp[i] = std::log(temp[i]);
        return _mm512_loadu_pd(temp);
    }
    static vector_type max(vector_type a, vector_type b) { return _mm512_max_pd(a, b); }
    static vector_type min(vector_type a, vector_type b) { return _mm512_min_pd(a, b); }
    static vector_type set1(double val) { return _mm512_set1_pd(val); }
};
#endif

// SIMD-aware vector operations
template<typename T, SimdLevel Level = SimdLevel::NONE>
class SimdVectorOps {
public:
    using Ops = SimdOps<T, Level>;
    static constexpr int width = Ops::vector_width;

    // Vector addition with scalar: result = a + alpha * b
    static void add_scaled(std::vector<T>& result, const std::vector<T>& a,
                          const std::vector<T>& b, T alpha) {
        size_t n = std::min({result.size(), a.size(), b.size()});
        size_t vec_end = (n / width) * width;

        // Vectorized loop
        for (size_t i = 0; i < vec_end; i += width) {
            auto va = Ops::load(&a[i]);
            auto vb = Ops::load(&b[i]);
            typename Ops::vector_type valpha;
            if constexpr (Level == SimdLevel::AVX2) {
#ifdef __AVX2__
                valpha = _mm256_set1_pd(alpha);
#endif
            } else if constexpr (Level == SimdLevel::AVX512) {
#ifdef __AVX512F__
                valpha = _mm512_set1_pd(alpha);
#endif
            } else {
                valpha = alpha;
            }
            auto vscaled = Ops::mul(valpha, vb);
            auto vresult = Ops::add(va, vscaled);
            Ops::store(&result[i], vresult);
        }

        // Scalar remainder
        for (size_t i = vec_end; i < n; ++i) {
            result[i] = a[i] + alpha * b[i];
        }
    }

    // Vector absolute value
    static void abs(std::vector<T>& result, const std::vector<T>& input) {
        size_t n = std::min(result.size(), input.size());
        size_t vec_end = (n / width) * width;

        for (size_t i = 0; i < vec_end; i += width) {
            auto vinput = Ops::load(&input[i]);
            auto vresult = Ops::abs(vinput);
            Ops::store(&result[i], vresult);
        }

        for (size_t i = vec_end; i < n; ++i) {
            result[i] = std::abs(input[i]);
        }
    }

    // Vector log (with safety check)
    static void log_safe(std::vector<T>& result, const std::vector<T>& input, T min_val) {
        size_t n = std::min(result.size(), input.size());
        size_t vec_end = (n / width) * width;

        for (size_t i = 0; i < vec_end; i += width) {
            auto vinput = Ops::load(&input[i]);
            typename Ops::vector_type vmin_val;
            if constexpr (Level == SimdLevel::AVX2) {
#ifdef __AVX2__
                vmin_val = _mm256_set1_pd(min_val);
#endif
            } else if constexpr (Level == SimdLevel::AVX512) {
#ifdef __AVX512F__
                vmin_val = _mm512_set1_pd(min_val);
#endif
            } else {
                vmin_val = min_val;
            }
            auto vsafe = Ops::max(vinput, vmin_val);
            auto vresult = Ops::log(vsafe);
            Ops::store(&result[i], vresult);
        }

        for (size_t i = vec_end; i < n; ++i) {
            result[i] = std::log(std::max(input[i], min_val));
        }
    }

    // Horizontal sum of vector elements
    static T horizontal_sum(const std::vector<T>& input) {
        size_t n = input.size();
        size_t vec_end = (n / width) * width;
        T sum = 0.0;

        // Vectorized sum
        for (size_t i = 0; i < vec_end; i += width) {
            auto vinput = Ops::load(&input[i]);
            // For simplicity, sum vector elements by extracting them
            // In practice, you might want more optimized horizontal operations
            if constexpr (Level == SimdLevel::AVX2) {
#ifdef __AVX2__
                double temp[4];
                _mm256_storeu_pd(temp, vinput);
                sum += temp[0] + temp[1] + temp[2] + temp[3];
#endif
            } else if constexpr (Level == SimdLevel::AVX512) {
#ifdef __AVX512F__
                double temp[8];
                _mm512_storeu_pd(temp, vinput);
                sum += temp[0] + temp[1] + temp[2] + temp[3] +
                       temp[4] + temp[5] + temp[6] + temp[7];
#endif
            } else {
                sum += input[i];
            }
        }

        // Scalar remainder
        for (size_t i = vec_end; i < n; ++i) {
            sum += input[i];
        }

        return sum;
    }

private:
    // Helper to set all elements to same value
    static typename Ops::vector_type set1(T val) {
        if constexpr (Level == SimdLevel::AVX2) {
            return _mm256_set1_pd(val);
        } else if constexpr (Level == SimdLevel::AVX512) {
            return _mm512_set1_pd(val);
        } else {
            return val;
        }
    }
};

// Runtime SIMD dispatcher
template<typename T>
class SimdDispatcher {
public:
    static SimdLevel get_level() {
        static SimdLevel level = SimdDetector::detect_capability();
        return level;
    }

    static void add_scaled(std::vector<T>& result, const std::vector<T>& a,
                          const std::vector<T>& b, T alpha, SimdLevel level) {
        switch (level) {
            case SimdLevel::AVX512:
#ifdef __AVX512F__
                SimdVectorOps<T, SimdLevel::AVX512>::add_scaled(result, a, b, alpha);
                break;
#endif
            case SimdLevel::AVX2:
#ifdef __AVX2__
                SimdVectorOps<T, SimdLevel::AVX2>::add_scaled(result, a, b, alpha);
                break;
#endif
            default:
                SimdVectorOps<T, SimdLevel::NONE>::add_scaled(result, a, b, alpha);
                break;
        }
    }

    static void abs(std::vector<T>& result, const std::vector<T>& input, SimdLevel level) {
        switch (level) {
            case SimdLevel::AVX512:
#ifdef __AVX512F__
                SimdVectorOps<T, SimdLevel::AVX512>::abs(result, input);
                break;
#endif
            case SimdLevel::AVX2:
#ifdef __AVX2__
                SimdVectorOps<T, SimdLevel::AVX2>::abs(result, input);
                break;
#endif
            default:
                SimdVectorOps<T, SimdLevel::NONE>::abs(result, input);
                break;
        }
    }

    static void log_safe(std::vector<T>& result, const std::vector<T>& input,
                        T min_val, SimdLevel level) {
        switch (level) {
            case SimdLevel::AVX512:
#ifdef __AVX512F__
                SimdVectorOps<T, SimdLevel::AVX512>::log_safe(result, input, min_val);
                break;
#endif
            case SimdLevel::AVX2:
#ifdef __AVX2__
                SimdVectorOps<T, SimdLevel::AVX2>::log_safe(result, input, min_val);
                break;
#endif
            default:
                SimdVectorOps<T, SimdLevel::NONE>::log_safe(result, input, min_val);
                break;
        }
    }

    static T horizontal_sum(const std::vector<T>& input, SimdLevel level) {
        switch (level) {
            case SimdLevel::AVX512:
#ifdef __AVX512F__
                return SimdVectorOps<T, SimdLevel::AVX512>::horizontal_sum(input);
#endif
            case SimdLevel::AVX2:
#ifdef __AVX2__
                return SimdVectorOps<T, SimdLevel::AVX2>::horizontal_sum(input);
#endif
            default:
                return SimdVectorOps<T, SimdLevel::NONE>::horizontal_sum(input);
        }
    }
};

} // namespace minisolver