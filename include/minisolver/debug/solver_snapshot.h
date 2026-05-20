#pragma once

#include "minisolver/algorithms/model_evaluation.h"
#include "minisolver/core/config_validation.h"
#include "minisolver/core/model_traits.h"
#include "minisolver/core/types.h"
#include "minisolver/solver/solver.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

namespace minisolver {

enum class SnapshotStatus {
    OK = 0,
    FileOpenFailed,
    UnsupportedVersion,
    DimensionMismatch,
    ModelMismatch,
    HorizonTooLarge,
    TruncatedFile,
    TrailingBytes,
    InvalidConfig,
    InvalidSnapshot,
    NonFiniteData,
    IoError
};

inline const char* snapshot_status_to_string(SnapshotStatus status)
{
    switch (status) {
    case SnapshotStatus::OK:
        return "OK";
    case SnapshotStatus::FileOpenFailed:
        return "FileOpenFailed";
    case SnapshotStatus::UnsupportedVersion:
        return "UnsupportedVersion";
    case SnapshotStatus::DimensionMismatch:
        return "DimensionMismatch";
    case SnapshotStatus::ModelMismatch:
        return "ModelMismatch";
    case SnapshotStatus::HorizonTooLarge:
        return "HorizonTooLarge";
    case SnapshotStatus::TruncatedFile:
        return "TruncatedFile";
    case SnapshotStatus::TrailingBytes:
        return "TrailingBytes";
    case SnapshotStatus::InvalidConfig:
        return "InvalidConfig";
    case SnapshotStatus::InvalidSnapshot:
        return "InvalidSnapshot";
    case SnapshotStatus::NonFiniteData:
        return "NonFiniteData";
    case SnapshotStatus::IoError:
        return "IoError";
    default:
        return "UNKNOWN";
    }
}

struct SnapshotResult {
    SnapshotStatus status = SnapshotStatus::OK;
    ApiStatus api_status = ApiStatus::OK;
    const char* message = "OK";

    explicit operator bool() const { return status == SnapshotStatus::OK; }
};

enum class SnapshotBackendPolicy { KeepConstructedBackend, UseSnapshotBackend, OverrideWith };

struct SnapshotLoadOptions {
    SnapshotBackendPolicy backend_policy = SnapshotBackendPolicy::KeepConstructedBackend;
    Backend override_backend = Backend::CPU_SERIAL;
    bool reject_trailing_bytes = true;
    bool reject_model_mismatch = true;
};

#define MINISOLVER_SNAPSHOT_CONFIG_FIELDS(X_ENUM, X_INT, X_DOUBLE, X_BOOL)                         \
    X_ENUM(backend)                                                                                \
    X_ENUM(initialization)                                                                         \
    X_ENUM(warm_start_barrier)                                                                     \
    X_ENUM(warm_start_regularization)                                                              \
    X_ENUM(termination_profile)                                                                    \
    X_ENUM(constraint_scaling)                                                                     \
    X_ENUM(objective_scaling)                                                                      \
    X_ENUM(problem_scaling)                                                                        \
    X_DOUBLE(constraint_row_scale_min)                                                             \
    X_DOUBLE(constraint_row_scale_max)                                                             \
    X_DOUBLE(objective_scale_min)                                                                  \
    X_DOUBLE(objective_scale_max)                                                                  \
    X_ENUM(integrator)                                                                             \
    X_DOUBLE(default_dt)                                                                           \
    X_INT(newton_config.max_iters)                                                                 \
    X_DOUBLE(newton_config.tol)                                                                    \
    X_DOUBLE(newton_config.regularization)                                                         \
    X_ENUM(barrier_strategy)                                                                       \
    X_DOUBLE(mu_init)                                                                              \
    X_DOUBLE(mu_final)                                                                             \
    X_DOUBLE(mu_linear_decrease_factor)                                                            \
    X_DOUBLE(barrier_tolerance_factor)                                                             \
    X_DOUBLE(mu_safety_margin)                                                                     \
    X_ENUM(inertia_strategy)                                                                       \
    X_DOUBLE(reg_init)                                                                             \
    X_DOUBLE(reg_min)                                                                              \
    X_DOUBLE(reg_max)                                                                              \
    X_DOUBLE(reg_scale_up)                                                                         \
    X_DOUBLE(reg_scale_down)                                                                       \
    X_DOUBLE(regularization_step)                                                                  \
    X_DOUBLE(singular_threshold)                                                                   \
    X_DOUBLE(huge_penalty)                                                                         \
    X_INT(linear_solve_max_attempts)                                                               \
    X_DOUBLE(tol_con)                                                                              \
    X_DOUBLE(tol_dual)                                                                             \
    X_DOUBLE(tol_mu)                                                                               \
    X_DOUBLE(tol_cost)                                                                             \
    X_DOUBLE(feasible_tol_scale)                                                                   \
    X_BOOL(enable_residual_stagnation_detection)                                                   \
    X_INT(residual_stagnation_min_iters)                                                           \
    X_INT(residual_stagnation_window)                                                              \
    X_DOUBLE(residual_stagnation_rel_tol)                                                          \
    X_DOUBLE(residual_stagnation_abs_tol)                                                          \
    X_ENUM(line_search_type)                                                                       \
    X_INT(line_search_max_iters)                                                                   \
    X_DOUBLE(line_search_tau)                                                                      \
    X_DOUBLE(line_search_backtrack_factor)                                                         \
    X_DOUBLE(filter_gamma_theta)                                                                   \
    X_DOUBLE(filter_gamma_phi)                                                                     \
    X_DOUBLE(filter_theta_max_factor)                                                              \
    X_DOUBLE(armijo_c1)                                                                            \
    X_DOUBLE(min_barrier_slack)                                                                    \
    X_DOUBLE(barrier_inf_cost)                                                                     \
    X_DOUBLE(slack_reset_trigger)                                                                  \
    X_DOUBLE(warm_start_slack_init)                                                                \
    X_DOUBLE(soc_trigger_alpha)                                                                    \
    X_DOUBLE(merit_nu_init)                                                                        \
    X_DOUBLE(eta_suff_descent)                                                                     \
    X_INT(max_restoration_iters)                                                                   \
    X_DOUBLE(restoration_mu)                                                                       \
    X_DOUBLE(restoration_reg)                                                                      \
    X_DOUBLE(restoration_alpha)                                                                    \
    X_DOUBLE(restoration_sufficient_decrease_factor)                                               \
    X_INT(max_iters)                                                                               \
    X_ENUM(print_level)                                                                            \
    X_BOOL(enable_profiling)                                                                       \
    X_ENUM(hessian_approximation)                                                                  \
    X_ENUM(direction_refinement)                                                                   \
    X_BOOL(enable_line_search_rollout)                                                             \
    X_BOOL(enable_defect_correction)                                                               \
    X_BOOL(enable_corrector)                                                                       \
    X_BOOL(enable_aggressive_barrier)                                                              \
    X_BOOL(enable_slack_reset)                                                                     \
    X_BOOL(enable_feasibility_restoration)                                                         \
    X_BOOL(enable_soc)

template <typename Model, int MAX_N> class SolverSnapshotIO {
public:
    using SolverType = MiniSolver<Model, MAX_N>;
    static constexpr std::array<char, 8> kMagic = { 'M', 'S', 'N', 'A', 'P', '0', '1', '\0' };
    static constexpr std::uint32_t kFormatVersion = 3;

    struct Snapshot {
        SolverConfig config;
        int N = 0;
        std::vector<double> dt_traj;

        SolverStatus status = SolverStatus::UNSOLVED;
        int iterations = 0;
        double total_cost = 0.0;
        double mu = 0.0;
        double reg = 0.0;

        struct KnotData {
            std::array<double, Model::NX> x;
            std::array<double, Model::NU> u;
            std::array<double, Model::NP> p;
            std::array<double, Model::NC> s;
            std::array<double, Model::NC> soft_s;
            std::array<double, Model::NC> lam;
        };
        std::vector<KnotData> trajectory;
    };

private:
    template <typename, typename = void> struct has_constraint_types : std::false_type { };
    template <typename T>
    struct has_constraint_types<T, std::void_t<decltype(T::constraint_types)>> : std::true_type { };

    template <typename, typename = void> struct has_constraint_weights : std::false_type { };
    template <typename T>
    struct has_constraint_weights<T, std::void_t<decltype(T::constraint_weights)>>
        : std::true_type { };

    static SnapshotResult result(SnapshotStatus status, ApiStatus api_status = ApiStatus::OK)
    {
        return { status, api_status, snapshot_status_to_string(status) };
    }

    template <typename T> static bool write_pod(std::ofstream& out, const T& value)
    {
        out.write(reinterpret_cast<const char*>(&value), static_cast<std::streamsize>(sizeof(T)));
        return out.good();
    }

    template <typename T> static bool read_pod(std::ifstream& in, T& value)
    {
        in.read(reinterpret_cast<char*>(&value), static_cast<std::streamsize>(sizeof(T)));
        return in.good();
    }

    template <typename Enum> static bool write_enum(std::ofstream& out, Enum value)
    {
        const std::int32_t raw = static_cast<std::int32_t>(value);
        return write_pod(out, raw);
    }

    template <typename Enum> static bool read_enum(std::ifstream& in, Enum& value)
    {
        std::int32_t raw = 0;
        if (!read_pod(in, raw)) {
            return false;
        }
        value = static_cast<Enum>(raw);
        return true;
    }

    static bool write_int(std::ofstream& out, int value)
    {
        const std::int32_t raw = static_cast<std::int32_t>(value);
        return write_pod(out, raw);
    }

    static bool read_int(std::ifstream& in, int& value)
    {
        std::int32_t raw = 0;
        if (!read_pod(in, raw)) {
            return false;
        }
        value = static_cast<int>(raw);
        return true;
    }

    static bool write_bool(std::ofstream& out, bool value)
    {
        const std::uint8_t raw = value ? 1u : 0u;
        return write_pod(out, raw);
    }

    static bool read_bool(std::ifstream& in, bool& value, bool& invalid_encoding)
    {
        std::uint8_t raw = 0;
        if (!read_pod(in, raw)) {
            return false;
        }
        if (raw > 1u) {
            invalid_encoding = true;
            return false;
        }
        value = (raw == 1u);
        return true;
    }

    static bool write_bytes(std::ofstream& out, const void* data, std::size_t bytes)
    {
        if (bytes == 0) {
            return true;
        }
        out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
        return out.good();
    }

    static bool read_bytes(std::ifstream& in, void* data, std::size_t bytes)
    {
        if (bytes == 0) {
            return true;
        }
        in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(bytes));
        return in.good();
    }

    static bool write_config(std::ofstream& out, const SolverConfig& cfg)
    {
        bool ok = true;
#define MS_SNAPSHOT_WRITE_ENUM(field) ok = ok && write_enum(out, cfg.field);
#define MS_SNAPSHOT_WRITE_INT(field) ok = ok && write_int(out, cfg.field);
#define MS_SNAPSHOT_WRITE_DOUBLE(field) ok = ok && write_pod(out, cfg.field);
#define MS_SNAPSHOT_WRITE_BOOL(field) ok = ok && write_bool(out, cfg.field);
        MINISOLVER_SNAPSHOT_CONFIG_FIELDS(MS_SNAPSHOT_WRITE_ENUM, MS_SNAPSHOT_WRITE_INT,
            MS_SNAPSHOT_WRITE_DOUBLE, MS_SNAPSHOT_WRITE_BOOL)
#undef MS_SNAPSHOT_WRITE_ENUM
#undef MS_SNAPSHOT_WRITE_INT
#undef MS_SNAPSHOT_WRITE_DOUBLE
#undef MS_SNAPSHOT_WRITE_BOOL
        return ok;
    }

    static bool read_config(std::ifstream& in, SolverConfig& cfg, bool& invalid_bool_encoding)
    {
        bool ok = true;
#define MS_SNAPSHOT_READ_ENUM(field) ok = ok && read_enum(in, cfg.field);
#define MS_SNAPSHOT_READ_INT(field) ok = ok && read_int(in, cfg.field);
#define MS_SNAPSHOT_READ_DOUBLE(field) ok = ok && read_pod(in, cfg.field);
#define MS_SNAPSHOT_READ_BOOL(field) ok = ok && read_bool(in, cfg.field, invalid_bool_encoding);
        MINISOLVER_SNAPSHOT_CONFIG_FIELDS(MS_SNAPSHOT_READ_ENUM, MS_SNAPSHOT_READ_INT,
            MS_SNAPSHOT_READ_DOUBLE, MS_SNAPSHOT_READ_BOOL)
#undef MS_SNAPSHOT_READ_ENUM
#undef MS_SNAPSHOT_READ_INT
#undef MS_SNAPSHOT_READ_DOUBLE
#undef MS_SNAPSHOT_READ_BOOL
        return ok;
    }

    static void hash_byte(std::uint64_t& hash, std::uint8_t value)
    {
        hash ^= static_cast<std::uint64_t>(value);
        hash *= 1099511628211ull;
    }

    template <typename T> static void hash_integral(std::uint64_t& hash, T value)
    {
        using U = std::uint64_t;
        U raw = static_cast<U>(value);
        for (std::size_t i = 0; i < sizeof(T); ++i) {
            hash_byte(hash, static_cast<std::uint8_t>(raw & 0xffu));
            raw >>= 8;
        }
    }

    static void hash_double(std::uint64_t& hash, double value)
    {
        std::uint64_t raw = 0;
        std::memcpy(&raw, &value, sizeof(double));
        hash_integral(hash, raw);
    }

    static void hash_string(std::uint64_t& hash, const char* value)
    {
        if (value == nullptr) {
            hash_byte(hash, 0xffu);
            return;
        }
        while (*value != '\0') {
            hash_byte(hash, static_cast<std::uint8_t>(*value));
            ++value;
        }
        hash_byte(hash, 0u);
    }

    template <typename Array> static void hash_name_array(std::uint64_t& hash, const Array& names)
    {
        hash_integral(hash, static_cast<std::uint64_t>(names.size()));
        for (const char* name : names) {
            hash_string(hash, name);
        }
    }

    static std::uint64_t model_fingerprint()
    {
        std::uint64_t hash = 1469598103934665603ull;
        hash_integral(hash, Model::NX);
        hash_integral(hash, Model::NU);
        hash_integral(hash, Model::NP);
        hash_integral(hash, Model::NC);
        hash_name_array(hash, Model::state_names);
        hash_name_array(hash, Model::control_names);
        hash_name_array(hash, Model::param_names);
        if constexpr (has_constraint_types<Model>::value) {
            hash_integral(hash, static_cast<std::uint64_t>(Model::constraint_types.size()));
            for (int type : Model::constraint_types) {
                hash_integral(hash, type);
            }
        } else {
            hash_string(hash, "no_constraint_types");
        }
        if constexpr (has_constraint_weights<Model>::value) {
            hash_integral(hash, static_cast<std::uint64_t>(Model::constraint_weights.size()));
            for (double weight : Model::constraint_weights) {
                hash_double(hash, weight);
            }
        } else {
            hash_string(hash, "no_constraint_weights");
        }
        if constexpr (detail::has_generated_integrator_v<Model>) {
            hash_integral(hash, static_cast<int>(Model::generated_integrator));
        }
        if constexpr (detail::has_model_fingerprint_v<Model>) {
            hash_integral(hash, Model::model_fingerprint);
        } else {
            hash_string(hash, "no_model_fingerprint");
        }
        return hash;
    }

    static bool finite_array(const double* data, std::size_t count)
    {
        for (std::size_t i = 0; i < count; ++i) {
            if (!std::isfinite(data[i])) {
                return false;
            }
        }
        return true;
    }

    static SnapshotStatus validate_snapshot_shape_and_data(const Snapshot& snapshot)
    {
        if (snapshot.N < 0) {
            return SnapshotStatus::InvalidSnapshot;
        }
        if (snapshot.N > MAX_N) {
            return SnapshotStatus::HorizonTooLarge;
        }
        if (static_cast<int>(snapshot.dt_traj.size()) != snapshot.N) {
            return SnapshotStatus::InvalidSnapshot;
        }
        if (static_cast<int>(snapshot.trajectory.size()) != snapshot.N + 1) {
            return SnapshotStatus::InvalidSnapshot;
        }
        if (detail::validate_solver_config(snapshot.config) != ApiStatus::OK) {
            return SnapshotStatus::InvalidConfig;
        }
        if (!std::isfinite(snapshot.total_cost) || !std::isfinite(snapshot.mu)
            || !std::isfinite(snapshot.reg)) {
            return SnapshotStatus::NonFiniteData;
        }
        if (!valid_solver_status(snapshot.status) || snapshot.iterations < 0 || snapshot.mu <= 0.0
            || snapshot.reg < 0.0) {
            return SnapshotStatus::InvalidSnapshot;
        }
        for (double dt : snapshot.dt_traj) {
            if (!std::isfinite(dt)) {
                return SnapshotStatus::NonFiniteData;
            }
        }
        for (const auto& knot : snapshot.trajectory) {
            if (!finite_array(knot.x.data(), knot.x.size())
                || !finite_array(knot.u.data(), knot.u.size())
                || !finite_array(knot.p.data(), knot.p.size())
                || !finite_array(knot.s.data(), knot.s.size())
                || !finite_array(knot.soft_s.data(), knot.soft_s.size())
                || !finite_array(knot.lam.data(), knot.lam.size())) {
                return SnapshotStatus::NonFiniteData;
            }
        }
        return SnapshotStatus::OK;
    }

    static SnapshotResult read_snapshot_file(
        const std::string& filename, const SnapshotLoadOptions& options, Snapshot& snapshot)
    {
        std::ifstream in(filename, std::ios::binary);
        if (!in) {
            return result(SnapshotStatus::FileOpenFailed);
        }

        std::array<char, 8> magic = {};
        if (!read_bytes(in, magic.data(), magic.size())) {
            return result(SnapshotStatus::TruncatedFile);
        }
        if (magic != kMagic) {
            return result(SnapshotStatus::UnsupportedVersion);
        }

        std::uint32_t version = 0;
        std::uint32_t scalar_bytes = 0;
        std::int32_t nx = 0;
        std::int32_t nu = 0;
        std::int32_t np = 0;
        std::int32_t nc = 0;
        std::int32_t horizon = 0;
        std::uint64_t fingerprint = 0;
        if (!read_pod(in, version) || !read_pod(in, scalar_bytes) || !read_pod(in, nx)
            || !read_pod(in, nu) || !read_pod(in, np) || !read_pod(in, nc) || !read_pod(in, horizon)
            || !read_pod(in, fingerprint)) {
            return result(SnapshotStatus::TruncatedFile);
        }

        if (version != kFormatVersion || scalar_bytes != sizeof(double)) {
            return result(SnapshotStatus::UnsupportedVersion);
        }
        if (nx != Model::NX || nu != Model::NU || np != Model::NP || nc != Model::NC) {
            return result(SnapshotStatus::DimensionMismatch);
        }
        if (horizon < 0) {
            return result(SnapshotStatus::InvalidSnapshot);
        }
        if (horizon > MAX_N) {
            return result(SnapshotStatus::HorizonTooLarge);
        }
        if (options.reject_model_mismatch && fingerprint != model_fingerprint()) {
            return result(SnapshotStatus::ModelMismatch);
        }

        snapshot.N = static_cast<int>(horizon);
        bool invalid_bool_encoding = false;
        if (!read_config(in, snapshot.config, invalid_bool_encoding)) {
            return result(invalid_bool_encoding ? SnapshotStatus::InvalidSnapshot
                                                : SnapshotStatus::TruncatedFile);
        }

        std::int32_t status_raw = 0;
        if (!read_pod(in, status_raw) || !read_int(in, snapshot.iterations)
            || !read_pod(in, snapshot.total_cost) || !read_pod(in, snapshot.mu)
            || !read_pod(in, snapshot.reg)) {
            return result(SnapshotStatus::TruncatedFile);
        }
        snapshot.status = static_cast<SolverStatus>(status_raw);
        if (!valid_solver_status(snapshot.status)) {
            return result(SnapshotStatus::InvalidSnapshot);
        }

        snapshot.dt_traj.assign(static_cast<std::size_t>(snapshot.N), 0.0);
        if (!read_bytes(in, snapshot.dt_traj.data(), sizeof(double) * snapshot.dt_traj.size())) {
            return result(SnapshotStatus::TruncatedFile);
        }

        snapshot.trajectory.assign(
            static_cast<std::size_t>(snapshot.N + 1), typename Snapshot::KnotData {});
        for (int k = 0; k <= snapshot.N; ++k) {
            auto& knot = snapshot.trajectory[static_cast<std::size_t>(k)];
            if (!read_bytes(in, knot.x.data(), sizeof(double) * knot.x.size())
                || !read_bytes(in, knot.u.data(), sizeof(double) * knot.u.size())
                || !read_bytes(in, knot.p.data(), sizeof(double) * knot.p.size())
                || !read_bytes(in, knot.s.data(), sizeof(double) * knot.s.size())
                || !read_bytes(in, knot.soft_s.data(), sizeof(double) * knot.soft_s.size())
                || !read_bytes(in, knot.lam.data(), sizeof(double) * knot.lam.size())) {
                return result(SnapshotStatus::TruncatedFile);
            }
        }

        if (options.reject_trailing_bytes && in.peek() != std::ifstream::traits_type::eof()) {
            return result(SnapshotStatus::TrailingBytes);
        }

        const SnapshotStatus validation = validate_snapshot_shape_and_data(snapshot);
        if (validation != SnapshotStatus::OK) {
            return result(validation);
        }
        return result(SnapshotStatus::OK);
    }

public:
    static bool config_equal(const SolverConfig& lhs, const SolverConfig& rhs)
    {
        bool equal = true;
#define MS_SNAPSHOT_CONFIG_EQ_ENUM(field) equal = equal && (lhs.field == rhs.field);
#define MS_SNAPSHOT_CONFIG_EQ_INT(field) equal = equal && (lhs.field == rhs.field);
#define MS_SNAPSHOT_CONFIG_EQ_DOUBLE(field) equal = equal && (lhs.field == rhs.field);
#define MS_SNAPSHOT_CONFIG_EQ_BOOL(field) equal = equal && (lhs.field == rhs.field);
        MINISOLVER_SNAPSHOT_CONFIG_FIELDS(MS_SNAPSHOT_CONFIG_EQ_ENUM, MS_SNAPSHOT_CONFIG_EQ_INT,
            MS_SNAPSHOT_CONFIG_EQ_DOUBLE, MS_SNAPSHOT_CONFIG_EQ_BOOL)
#undef MS_SNAPSHOT_CONFIG_EQ_ENUM
#undef MS_SNAPSHOT_CONFIG_EQ_INT
#undef MS_SNAPSHOT_CONFIG_EQ_DOUBLE
#undef MS_SNAPSHOT_CONFIG_EQ_BOOL
        return equal;
    }

    // Allocating debug capture. Do not call from hard real-time control loops.
    static Snapshot capture_snapshot(
        const SolverType& solver, SolverStatus status = SolverStatus::UNSOLVED)
    {
        Snapshot snapshot;
        snapshot.config = solver.config;
        snapshot.N = solver.N;
        snapshot.status = status;
        snapshot.iterations = solver.context_.solve.current_iter;
        snapshot.mu = solver.context_.solve.mu;
        snapshot.reg = solver.context_.solve.reg;

        snapshot.dt_traj.resize(static_cast<std::size_t>(solver.N));
        for (int k = 0; k < solver.N; ++k) {
            snapshot.dt_traj[static_cast<std::size_t>(k)] = solver.dt_traj[k];
        }

        snapshot.trajectory.resize(static_cast<std::size_t>(solver.N + 1));
        const auto& active = solver.trajectory.active();

        snapshot.total_cost = 0.0;
        for (int k = 0; k <= solver.N; ++k) {
            const auto& kp = active[k];
            auto& data = snapshot.trajectory[static_cast<std::size_t>(k)];
            snapshot.total_cost += kp.cost;
            for (int i = 0; i < Model::NX; ++i) {
                data.x[static_cast<std::size_t>(i)] = kp.x(i);
            }
            for (int i = 0; i < Model::NU; ++i) {
                data.u[static_cast<std::size_t>(i)] = kp.u(i);
            }
            for (int i = 0; i < Model::NP; ++i) {
                data.p[static_cast<std::size_t>(i)] = kp.p(i);
            }
            for (int i = 0; i < Model::NC; ++i) {
                data.s[static_cast<std::size_t>(i)] = kp.s(i);
            }
            for (int i = 0; i < Model::NC; ++i) {
                data.soft_s[static_cast<std::size_t>(i)] = kp.soft_s(i);
            }
            for (int i = 0; i < Model::NC; ++i) {
                data.lam[static_cast<std::size_t>(i)] = kp.lam(i);
            }
        }

        return snapshot;
    }

    static SnapshotResult save_snapshot(const std::string& filename, const Snapshot& snapshot)
    {
        const SnapshotStatus validation = validate_snapshot_shape_and_data(snapshot);
        if (validation != SnapshotStatus::OK) {
            return result(validation);
        }

        std::ofstream out(filename, std::ios::binary);
        if (!out) {
            return result(SnapshotStatus::FileOpenFailed);
        }

        const std::uint32_t version = kFormatVersion;
        const std::uint32_t scalar_bytes = sizeof(double);
        const std::int32_t nx = Model::NX;
        const std::int32_t nu = Model::NU;
        const std::int32_t np = Model::NP;
        const std::int32_t nc = Model::NC;
        const std::int32_t horizon = snapshot.N;
        const std::uint64_t fingerprint = model_fingerprint();

        bool ok = write_bytes(out, kMagic.data(), kMagic.size()) && write_pod(out, version)
            && write_pod(out, scalar_bytes) && write_pod(out, nx) && write_pod(out, nu)
            && write_pod(out, np) && write_pod(out, nc) && write_pod(out, horizon)
            && write_pod(out, fingerprint) && write_config(out, snapshot.config);

        const std::int32_t status_raw = static_cast<std::int32_t>(snapshot.status);
        ok = ok && write_pod(out, status_raw) && write_int(out, snapshot.iterations)
            && write_pod(out, snapshot.total_cost) && write_pod(out, snapshot.mu)
            && write_pod(out, snapshot.reg);

        ok = ok
            && write_bytes(out, snapshot.dt_traj.data(), sizeof(double) * snapshot.dt_traj.size());
        for (const auto& knot : snapshot.trajectory) {
            ok = ok && write_bytes(out, knot.x.data(), sizeof(double) * knot.x.size())
                && write_bytes(out, knot.u.data(), sizeof(double) * knot.u.size())
                && write_bytes(out, knot.p.data(), sizeof(double) * knot.p.size())
                && write_bytes(out, knot.s.data(), sizeof(double) * knot.s.size())
                && write_bytes(out, knot.soft_s.data(), sizeof(double) * knot.soft_s.size())
                && write_bytes(out, knot.lam.data(), sizeof(double) * knot.lam.size());
        }

        if (!ok || out.bad()) {
            return result(SnapshotStatus::IoError);
        }
        return result(SnapshotStatus::OK);
    }

    static bool is_failure_status(SolverStatus status)
    {
        return status != SolverStatus::OPTIMAL && status != SolverStatus::FEASIBLE;
    }

    // Persist the pre-solve replay state only when solve_status reports failure.
    // This keeps the explicit user flow:
    //   auto pre_solve = SnapshotIO::capture_snapshot(solver);
    //   SolverStatus status = solver.solve();
    //   SnapshotIO::save_failure_snapshot("failed.msnap", pre_solve, status);
    static SnapshotResult save_failure_snapshot(
        const std::string& filename, const Snapshot& pre_solve_snapshot, SolverStatus solve_status)
    {
        if (!is_failure_status(solve_status)) {
            return result(SnapshotStatus::OK);
        }

        Snapshot failure_snapshot = pre_solve_snapshot;
        failure_snapshot.status = solve_status;
        return save_snapshot(filename, failure_snapshot);
    }

    static SnapshotResult save_case(const std::string& filename, const SolverType& solver)
    {
        return save_snapshot(filename, capture_snapshot(solver));
    }

    static SnapshotResult load_case(
        const std::string& filename, SolverType& solver, SnapshotLoadOptions options = {})
    {
        Snapshot snapshot;
        SnapshotResult read_result = read_snapshot_file(filename, options, snapshot);
        if (!read_result) {
            return read_result;
        }

        SolverConfig config = snapshot.config;
        if (options.backend_policy == SnapshotBackendPolicy::KeepConstructedBackend) {
            config.backend = solver.config.backend;
        } else if (options.backend_policy == SnapshotBackendPolicy::OverrideWith) {
            config.backend = options.override_backend;
        }

        ApiStatus restore_status = solver.restore_config_from_snapshot_(config);
        if (restore_status != ApiStatus::OK) {
            return result(SnapshotStatus::InvalidConfig, restore_status);
        }
        restore_status = solver.resize_horizon(snapshot.N);
        if (restore_status != ApiStatus::OK) {
            return result(SnapshotStatus::InvalidSnapshot, restore_status);
        }

        solver.context_.solve.current_iter = snapshot.iterations;
        solver.context_.solve.mu = snapshot.mu;
        solver.context_.solve.reg = snapshot.reg;
        solver.build_state_.dirty = true;
        solver.rebuild_solver_components_if_dirty_();

        for (int k = 0; k < snapshot.N; ++k) {
            solver.dt_traj[k] = snapshot.dt_traj[static_cast<std::size_t>(k)];
        }

        auto& traj = solver.trajectory.active();
        for (int k = 0; k <= snapshot.N; ++k) {
            const auto& src = snapshot.trajectory[static_cast<std::size_t>(k)];
            for (int i = 0; i < Model::NX; ++i) {
                traj[k].x(i) = src.x[static_cast<std::size_t>(i)];
            }
            for (int i = 0; i < Model::NU; ++i) {
                traj[k].u(i) = src.u[static_cast<std::size_t>(i)];
            }
            for (int i = 0; i < Model::NP; ++i) {
                traj[k].p(i) = src.p[static_cast<std::size_t>(i)];
            }
            for (int i = 0; i < Model::NC; ++i) {
                traj[k].s(i) = src.s[static_cast<std::size_t>(i)];
            }
            for (int i = 0; i < Model::NC; ++i) {
                traj[k].soft_s(i) = src.soft_s[static_cast<std::size_t>(i)];
            }
            for (int i = 0; i < Model::NC; ++i) {
                traj[k].lam(i) = src.lam[static_cast<std::size_t>(i)];
            }
        }

        solver.trajectory.prepare_candidate();
        for (int k = 0; k <= snapshot.N; ++k) {
            const double current_dt = (k < snapshot.N) ? solver.dt_traj[k] : 0.0;
            detail::evaluate_model_stage<Model>(
                traj[k], solver.config, current_dt, k == snapshot.N);
        }

        return result(SnapshotStatus::OK);
    }
};

#undef MINISOLVER_SNAPSHOT_CONFIG_FIELDS

} // namespace minisolver
