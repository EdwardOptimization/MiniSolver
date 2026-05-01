#pragma once

#include "minisolver/matrix/policies.h"
#include "minisolver/matrix/static_for.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace minisolver {

template <typename T, int R, int C> class MiniMatrix;

namespace matrix {

    inline uint64_t double_to_bits(double val)
    {
        uint64_t bits;
        std::memcpy(&bits, &val, sizeof(bits));
        return bits;
    }

    inline bool is_nan_scalar(double val)
    {
        const uint64_t bits = double_to_bits(val);
        return ((bits & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL)
            && ((bits & 0x000FFFFFFFFFFFFFULL) != 0);
    }

    inline bool is_finite_scalar(double val)
    {
        const uint64_t bits = double_to_bits(val);
        return (bits & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
    }

    template <bool Unroll> struct FillImpl;

    template <> struct FillImpl<true> {
        template <typename Mat, typename T> static inline void run(Mat& m, T value)
        {
            struct Body {
                Mat& m;
                T value;
                inline void operator()(int i) { m.data[i] = value; }
            } body = { m, value };
            StaticFor<0, Mat::Rows * Mat::Cols>::run(body);
        }
    };

    template <> struct FillImpl<false> {
        template <typename Mat, typename T> static inline void run(Mat& m, T value)
        {
            for (int i = 0; i < Mat::Rows * Mat::Cols; ++i) {
                m.data[i] = value;
            }
        }
    };

    template <typename Mat, typename T> inline void fill(Mat& m, T value)
    {
        FillImpl<MatrixPolicy::StaticUnroll<Mat::Rows * Mat::Cols>::value>::run(m, value);
    }

    template <bool Unroll> struct ScaleImpl;

    template <> struct ScaleImpl<true> {
        template <typename Out, typename In, typename T>
        static inline void run(Out& out, const In& in, T scale)
        {
            struct Body {
                Out& out;
                const In& in;
                T scale;
                inline void operator()(int i) { out.data[i] = in.data[i] * scale; }
            } body = { out, in, scale };
            StaticFor<0, In::Rows * In::Cols>::run(body);
        }
    };

    template <> struct ScaleImpl<false> {
        template <typename Out, typename In, typename T>
        static inline void run(Out& out, const In& in, T scale)
        {
            for (int i = 0; i < In::Rows * In::Cols; ++i) {
                out.data[i] = in.data[i] * scale;
            }
        }
    };

    template <typename Out, typename In, typename T>
    inline void scale(Out& out, const In& in, T scale)
    {
        ScaleImpl<MatrixPolicy::StaticUnroll<In::Rows * In::Cols>::value>::run(out, in, scale);
    }

    template <bool Unroll, int Sign> struct AddSubImpl;

    template <int Sign> struct AddSubImpl<true, Sign> {
        template <typename Out, typename A, typename B>
        static inline void run(Out& out, const A& a, const B& b)
        {
            struct Body {
                Out& out;
                const A& a;
                const B& b;
                inline void operator()(int i) { out.data[i] = a.data[i] + Sign * b.data[i]; }
            } body = { out, a, b };
            StaticFor<0, A::Rows * A::Cols>::run(body);
        }
    };

    template <int Sign> struct AddSubImpl<false, Sign> {
        template <typename Out, typename A, typename B>
        static inline void run(Out& out, const A& a, const B& b)
        {
            for (int i = 0; i < A::Rows * A::Cols; ++i) {
                out.data[i] = a.data[i] + Sign * b.data[i];
            }
        }
    };

    template <typename Out, typename A, typename B>
    inline void add(Out& out, const A& a, const B& b)
    {
        AddSubImpl<MatrixPolicy::StaticUnroll<A::Rows * A::Cols>::value, 1>::run(out, a, b);
    }

    template <typename Out, typename A, typename B>
    inline void sub(Out& out, const A& a, const B& b)
    {
        AddSubImpl<MatrixPolicy::StaticUnroll<A::Rows * A::Cols>::value, -1>::run(out, a, b);
    }

    template <bool Unroll> struct AddScaledImpl;

    template <> struct AddScaledImpl<true> {
        template <typename Out, typename In, typename T>
        static inline void run(Out& out, const In& in, T scale)
        {
            struct Body {
                Out& out;
                const In& in;
                T scale;
                inline void operator()(int i) { out.data[i] += in.data[i] * scale; }
            } body = { out, in, scale };
            StaticFor<0, In::Rows * In::Cols>::run(body);
        }
    };

    template <> struct AddScaledImpl<false> {
        template <typename Out, typename In, typename T>
        static inline void run(Out& out, const In& in, T scale)
        {
            for (int i = 0; i < In::Rows * In::Cols; ++i) {
                out.data[i] += in.data[i] * scale;
            }
        }
    };

    template <typename Out, typename In, typename T>
    inline void add_scaled(Out& out, const In& in, T scale)
    {
        AddScaledImpl<MatrixPolicy::StaticUnroll<In::Rows * In::Cols>::value>::run(out, in, scale);
    }

    template <bool Unroll> struct MatMulImpl;

    template <> struct MatMulImpl<true> {
        template <typename Out, typename A, typename B>
        static inline void run(Out& out, const A& a, const B& b)
        {
            struct Body {
                Out& out;
                const A& a;
                const B& b;
                inline void operator()(int index)
                {
                    const int i = index / Out::Cols;
                    const int j = index - i * Out::Cols;
                    typename Out::Scalar sum = typename Out::Scalar(0);
                    for (int k = 0; k < A::Cols; ++k) {
                        sum += a(i, k) * b(k, j);
                    }
                    out(i, j) = sum;
                }
            } body = { out, a, b };
            StaticFor<0, Out::Rows * Out::Cols>::run(body);
        }
    };

    template <> struct MatMulImpl<false> {
        template <typename Out, typename A, typename B>
        static inline void run(Out& out, const A& a, const B& b)
        {
            for (int i = 0; i < Out::Rows; ++i) {
                for (int j = 0; j < Out::Cols; ++j) {
                    typename Out::Scalar sum = typename Out::Scalar(0);
                    for (int k = 0; k < A::Cols; ++k) {
                        sum += a(i, k) * b(k, j);
                    }
                    out(i, j) = sum;
                }
            }
        }
    };

    template <typename Out, typename A, typename B>
    inline void matmul(Out& out, const A& a, const B& b)
    {
        MatMulImpl<MatrixPolicy::StaticUnroll<Out::Rows * Out::Cols * A::Cols>::value>::run(
            out, a, b);
    }

    template <bool Unroll> struct MatMulAddImpl;

    template <> struct MatMulAddImpl<true> {
        template <typename Out, typename A, typename B>
        static inline void run(Out& out, const A& a, const B& b)
        {
            struct Body {
                Out& out;
                const A& a;
                const B& b;
                inline void operator()(int index)
                {
                    const int i = index / Out::Cols;
                    const int j = index - i * Out::Cols;
                    typename Out::Scalar sum = typename Out::Scalar(0);
                    for (int k = 0; k < A::Cols; ++k) {
                        sum += a(i, k) * b(k, j);
                    }
                    out(i, j) += sum;
                }
            } body = { out, a, b };
            StaticFor<0, Out::Rows * Out::Cols>::run(body);
        }
    };

    template <> struct MatMulAddImpl<false> {
        template <typename Out, typename A, typename B>
        static inline void run(Out& out, const A& a, const B& b)
        {
            for (int i = 0; i < Out::Rows; ++i) {
                for (int j = 0; j < Out::Cols; ++j) {
                    typename Out::Scalar sum = typename Out::Scalar(0);
                    for (int k = 0; k < A::Cols; ++k) {
                        sum += a(i, k) * b(k, j);
                    }
                    out(i, j) += sum;
                }
            }
        }
    };

    template <typename Out, typename A, typename B>
    inline void matmul_add(Out& out, const A& a, const B& b)
    {
        MatMulAddImpl<MatrixPolicy::StaticUnroll<Out::Rows * Out::Cols * A::Cols>::value>::run(
            out, a, b);
    }

    template <bool Unroll> struct TransposeImpl;

    template <> struct TransposeImpl<true> {
        template <typename Out, typename In> static inline void run(Out& out, const In& in)
        {
            struct Body {
                Out& out;
                const In& in;
                inline void operator()(int index)
                {
                    const int i = index / In::Cols;
                    const int j = index - i * In::Cols;
                    out(j, i) = in(i, j);
                }
            } body = { out, in };
            StaticFor<0, In::Rows * In::Cols>::run(body);
        }
    };

    template <> struct TransposeImpl<false> {
        template <typename Out, typename In> static inline void run(Out& out, const In& in)
        {
            for (int i = 0; i < In::Rows; ++i) {
                for (int j = 0; j < In::Cols; ++j) {
                    out(j, i) = in(i, j);
                }
            }
        }
    };

    template <typename Out, typename In> inline void transpose(Out& out, const In& in)
    {
        TransposeImpl<MatrixPolicy::StaticUnroll<In::Rows * In::Cols>::value>::run(out, in);
    }

    template <bool Unroll> struct AddAtMulBImpl;

    template <> struct AddAtMulBImpl<true> {
        template <typename Out, typename A, typename B>
        static inline void run(Out& out, const A& a, const B& b)
        {
            struct Body {
                Out& out;
                const A& a;
                const B& b;
                inline void operator()(int index)
                {
                    const int i = index / Out::Cols;
                    const int j = index - i * Out::Cols;
                    typename Out::Scalar sum = typename Out::Scalar(0);
                    for (int k = 0; k < A::Rows; ++k) {
                        sum += a(k, i) * b(k, j);
                    }
                    out(i, j) += sum;
                }
            } body = { out, a, b };
            StaticFor<0, Out::Rows * Out::Cols>::run(body);
        }
    };

    template <> struct AddAtMulBImpl<false> {
        template <typename Out, typename A, typename B>
        static inline void run(Out& out, const A& a, const B& b)
        {
            for (int i = 0; i < Out::Rows; ++i) {
                for (int j = 0; j < Out::Cols; ++j) {
                    typename Out::Scalar sum = typename Out::Scalar(0);
                    for (int k = 0; k < A::Rows; ++k) {
                        sum += a(k, i) * b(k, j);
                    }
                    out(i, j) += sum;
                }
            }
        }
    };

    template <typename Out, typename A, typename B>
    inline void add_At_mul_B(Out& out, const A& a, const B& b)
    {
        AddAtMulBImpl<MatrixPolicy::StaticUnroll<Out::Rows * Out::Cols * A::Rows>::value>::run(
            out, a, b);
    }

    template <bool Unroll> struct WeightedAddAtMulBImpl;

    template <> struct WeightedAddAtMulBImpl<true> {
        template <typename Out, typename A, typename Weights, typename B>
        static inline void run(Out& out, const A& a, const Weights& weights, const B& b)
        {
            struct Body {
                Out& out;
                const A& a;
                const Weights& weights;
                const B& b;
                inline void operator()(int index)
                {
                    const int i = index / Out::Cols;
                    const int j = index - i * Out::Cols;
                    typename Out::Scalar sum = typename Out::Scalar(0);
                    for (int k = 0; k < A::Rows; ++k) {
                        sum += a(k, i) * weights(k) * b(k, j);
                    }
                    out(i, j) += sum;
                }
            } body = { out, a, weights, b };
            StaticFor<0, Out::Rows * Out::Cols>::run(body);
        }
    };

    template <> struct WeightedAddAtMulBImpl<false> {
        template <typename Out, typename A, typename Weights, typename B>
        static inline void run(Out& out, const A& a, const Weights& weights, const B& b)
        {
            for (int i = 0; i < Out::Rows; ++i) {
                for (int j = 0; j < Out::Cols; ++j) {
                    typename Out::Scalar sum = typename Out::Scalar(0);
                    for (int k = 0; k < A::Rows; ++k) {
                        sum += a(k, i) * weights(k) * b(k, j);
                    }
                    out(i, j) += sum;
                }
            }
        }
    };

    template <typename Out, typename A, typename Weights, typename B>
    inline void weighted_add_At_mul_B(Out& out, const A& a, const Weights& weights, const B& b)
    {
        WeightedAddAtMulBImpl<
            MatrixPolicy::StaticUnroll<Out::Rows * Out::Cols * A::Rows>::value>::run(out, a,
            weights, b);
    }

    template <bool Unroll> struct AddAtMulVImpl;

    template <> struct AddAtMulVImpl<true> {
        template <typename Out, typename A, typename X>
        static inline void run(Out& out, const A& a, const X& x)
        {
            struct Body {
                Out& out;
                const A& a;
                const X& x;
                inline void operator()(int i)
                {
                    typename Out::Scalar sum = typename Out::Scalar(0);
                    for (int k = 0; k < A::Rows; ++k) {
                        sum += a(k, i) * x(k);
                    }
                    out.data[i] += sum;
                }
            } body = { out, a, x };
            StaticFor<0, Out::Rows>::run(body);
        }
    };

    template <> struct AddAtMulVImpl<false> {
        template <typename Out, typename A, typename X>
        static inline void run(Out& out, const A& a, const X& x)
        {
            for (int i = 0; i < Out::Rows; ++i) {
                typename Out::Scalar sum = typename Out::Scalar(0);
                for (int k = 0; k < A::Rows; ++k) {
                    sum += a(k, i) * x(k);
                }
                out.data[i] += sum;
            }
        }
    };

    template <typename Out, typename A, typename X>
    inline void add_At_mul_v(Out& out, const A& a, const X& x)
    {
        AddAtMulVImpl<MatrixPolicy::StaticUnroll<Out::Rows * A::Rows>::value>::run(out, a, x);
    }

    template <bool Unroll> struct SymmetrizeImpl;

    template <> struct SymmetrizeImpl<true> {
        template <typename Mat> static inline void run(Mat& m)
        {
            struct Body {
                Mat& m;
                inline void operator()(int index)
                {
                    const int i = index / Mat::Cols;
                    const int j = index - i * Mat::Cols;
                    if (j > i) {
                        typename Mat::Scalar val = (m(i, j) + m(j, i)) * typename Mat::Scalar(0.5);
                        m(i, j) = val;
                        m(j, i) = val;
                    }
                }
            } body = { m };
            StaticFor<0, Mat::Rows * Mat::Cols>::run(body);
        }
    };

    template <> struct SymmetrizeImpl<false> {
        template <typename Mat> static inline void run(Mat& m)
        {
            for (int i = 0; i < Mat::Rows - 1; ++i) {
                for (int j = i + 1; j < Mat::Cols; ++j) {
                    typename Mat::Scalar val = (m(i, j) + m(j, i)) * typename Mat::Scalar(0.5);
                    m(i, j) = val;
                    m(j, i) = val;
                }
            }
        }
    };

    template <typename Mat> inline void symmetrize(Mat& m)
    {
        SymmetrizeImpl<MatrixPolicy::StaticUnroll<Mat::Rows * Mat::Cols>::value>::run(m);
    }

    template <bool Unroll> struct HasNanImpl;

    template <> struct HasNanImpl<true> {
        template <typename Mat> static inline bool run(const Mat& m)
        {
            for (int i = 0; i < Mat::Rows * Mat::Cols; ++i) {
                if (is_nan_scalar(static_cast<double>(m.data[i]))) {
                    return true;
                }
            }
            return false;
        }
    };

    template <> struct HasNanImpl<false> {
        template <typename Mat> static inline bool run(const Mat& m)
        {
            for (int i = 0; i < Mat::Rows; ++i) {
                for (int j = 0; j < Mat::Cols; ++j) {
                    if (is_nan_scalar(static_cast<double>(m(i, j)))) {
                        return true;
                    }
                }
            }
            return false;
        }
    };

    template <typename Mat> inline bool has_nan(const Mat& m)
    {
        return HasNanImpl<MatrixPolicy::StaticUnroll<Mat::Rows * Mat::Cols>::value>::run(m);
    }

    template <typename Mat> inline bool all_finite(const Mat& m)
    {
        for (int i = 0; i < Mat::Rows * Mat::Cols; ++i) {
            if (!is_finite_scalar(static_cast<double>(m.data[i]))) {
                return false;
            }
        }
        return true;
    }

    template <typename A, typename B> inline double dot(const A& a, const B& b)
    {
        double sum = 0.0;
        for (int i = 0; i < A::Rows * A::Cols; ++i) {
            sum += static_cast<double>(a.data[i]) * static_cast<double>(b.data[i]);
        }
        return sum;
    }

    template <typename Mat> inline double norm_inf(const Mat& m)
    {
        double max_val = 0.0;
        for (int i = 0; i < Mat::Rows * Mat::Cols; ++i) {
            max_val = std::max(max_val, std::abs(static_cast<double>(m.data[i])));
        }
        return max_val;
    }

}
}
