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

    template <int Work> struct UseStaticUnroll {
        enum { value = (Work <= 256) };
    };

}
}
