#pragma once

namespace minisolver {
namespace matrix {

    template <int Index, int End> struct StaticFor {
        template <typename Functor> static inline void run(Functor& f)
        {
            f(Index);
            StaticFor<Index + 1, End>::run(f);
        }
    };

    template <int End> struct StaticFor<End, End> {
        template <typename Functor> static inline void run(Functor&) { }
    };

    template <bool Unroll, int End> struct ForRange {
        template <typename Functor> static inline void run(Functor& f)
        {
            for (int i = 0; i < End; ++i)
                f(i);
        }
    };

    template <int End> struct ForRange<true, End> {
        template <typename Functor> static inline void run(Functor& f)
        {
            StaticFor<0, End>::run(f);
        }
    };

    template <bool Unroll, int End> struct PrefixRange {
        template <typename Functor> static inline void run(int count, Functor& f)
        {
            for (int i = 0; i < count; ++i)
                f(i);
        }
    };

    template <int End> struct PrefixRange<true, End> {
        template <typename Functor> static inline void run(int count, Functor& f)
        {
            struct Body {
                int count;
                Functor& f;
                inline void operator()(int i)
                {
                    if (i < count)
                        f(i);
                }
            } body = { count, f };
            StaticFor<0, End>::run(body);
        }
    };

    template <bool Unroll, int End> struct SuffixRange {
        template <typename Functor> static inline void run(int begin, Functor& f)
        {
            for (int i = begin; i < End; ++i)
                f(i);
        }
    };

    template <int End> struct SuffixRange<true, End> {
        template <typename Functor> static inline void run(int begin, Functor& f)
        {
            struct Body {
                int begin;
                Functor& f;
                inline void operator()(int i)
                {
                    if (i >= begin)
                        f(i);
                }
            } body = { begin, f };
            StaticFor<0, End>::run(body);
        }
    };

    template <int Work> struct UseStaticUnroll {
        enum { value = (Work <= 256) };
    };

}
}
