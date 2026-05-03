#pragma once
#include <array>

namespace minisolver::testdata {

struct OpenLoopReference5x2 {
    double objective;
    std::array<double, 5> terminal_state;
    std::array<double, 2> first_control;
};

struct OpenLoopReference6x3 {
    double objective;
    std::array<double, 6> terminal_state;
    std::array<double, 3> first_control;
};

inline constexpr OpenLoopReference5x2 kKinematicBicycleStraightReference {
    13.838830752626082,
    { 2.374838322011422, 0.018181507588070366, -0.0015864364214292473, 2.0613407869230085,
        0.026263409471181293 },
    { 4.0000000099902442, -2.5000000099586148 },
};

inline constexpr std::array<double, 5> kKinematicBicycleCurvedClosedLoopFinalState {
    1.009628953772298, 0.21997366992892475, -0.11077770731202181, 2.0098727383104911,
    0.14556064188638601
};

inline constexpr OpenLoopReference6x3 kDoubleIntegrator3DTrackingReference {
    87.20394461634848,
    { 0.97687468770877806, 0.1592328843341117, 1.0405490562345703, 0.40210319827008223,
        0.027493991474230827, -0.048444493889525077 },
    { 11.626077699170926, -1.880420005899418, 4.0108229571217286 },
};

inline constexpr OpenLoopReference6x3 kDoubleIntegrator3DShiftedReference {
    70.0492959884985,
    { 1.0798265707411723, 0.18294616666235819, 1.0337284417920172, 0.38545421056083279,
        0.0024949430437185751, -0.073565458489216445 },
    { 10.299772521224844, -1.789423529534528, 3.4489879228915585 },
};

} // namespace minisolver::testdata
