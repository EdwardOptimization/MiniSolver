import tempfile

from common import OptimalControlModel, compile_and_run


def test_generated_terminal_stage_uses_x_only_projection():
    model = OptimalControlModel("TerminalProjectionModel")
    x = model.state("x")
    u = model.control("u")
    model.set_dynamics(x, u)
    model.minimize((x + u) ** 2 + 3.0 * u**2)
    model.subject_to(x + 2.0 * u - 4.0 <= 0)
    model.subject_to(u - 1.0 <= 0)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.generate(tmpdir, integrator_type="EULER_EXPLICIT")
        compile_and_run(
            tmpdir,
            "terminal_projection_check.cpp",
            "terminal_projection_check",
            """
            #include "terminalprojectionmodel.h"
            #include <cmath>
            #include <cstdlib>

            int main() {
                using Model = minisolver::TerminalProjectionModel;
                minisolver::KnotPoint<double, Model::NX, Model::NU, Model::NC, Model::NP> kp;
                kp.set_zero();
                kp.x(0) = 3.0;
                kp.u(0) = 10.0;
                Model::compute_terminal_cost_exact(kp);
                Model::compute_terminal_qp_constraints(kp);
                if (std::abs(kp.cost - 9.0) > 1e-12) return 1;
                if (std::abs(kp.q(0) - 6.0) > 1e-12) return 2;
                if (std::abs(kp.r(0)) > 1e-12) return 3;
                if (std::abs(kp.g_val(0) - (-1.0)) > 1e-12) return 4;
                if (std::abs(kp.D(0,0)) > 1e-12) return 5;
                if (std::abs(kp.g_val(1) - (-1.0)) > 1e-12) return 6;
                return 0;
            }
            """,
        )


if __name__ == "__main__":
    test_generated_terminal_stage_uses_x_only_projection()
