from sympy import Symbol, Eq, Heaviside, sqrt
from sympy.logic.boolalg import Or
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags

import modulus.sym
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_2d import Rectangle
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    PointwiseConstraint,
)
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.inferencer import VoxelInferencer
from modulus.sym.utils.io.plotter import ValidatorPlotter, InferencerPlotter
from modulus.sym.key import Key
from modulus.sym.eq.pdes.maxwell import Maxwell2D
from modulus.sym.eq.pdes.navier_stokes import GradNormal

x, y = Symbol("x"), Symbol("y")

# helper function for computing laplacian eigen values
def Laplacian_1D_eig(a, b, N, eps=lambda x: np.ones_like(x), k=3):
    n = N - 2
    h = (b - a) / (N - 1)

    L = diags([1, -2, 1], [-1, 0, 1], shape=(n, n))
    L = -L / h**2

    x = np.linspace(a, b, num=N)
    M = diags([eps(x[1:-1])], [0])

    eigvals, eigvecs = eigsh(L, k=k, M=M, which="SM")
    eigvecs = np.vstack((np.zeros((1, k)), eigvecs, np.zeros((1, k))))
    norm_eigvecs = np.linalg.norm(eigvecs, axis=0)
    eigvecs /= norm_eigvecs
    return eigvals.astype(np.float32), eigvecs.astype(np.float32), x.astype(np.float32)


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # params for domain
    height = 2
    width = 2

    len_slab = 0.6
    eps0 = 1.0
    eps1 = 2.0
    eps_numpy = lambda y: np.where(
        np.logical_and(y > (height - len_slab) / 2, y < (height + len_slab) / 2),
        eps1,
        eps0,
    )
    eps_sympy = sqrt(
        eps0
        + (
            Heaviside(y - (height - len_slab) / 2)
            - Heaviside(y - (height + len_slab) / 2)
        )
        * (eps1 - eps0)
    )
    eigvals, eigvecs, yv = Laplacian_1D_eig(0, height, 1000, eps=eps_numpy, k=3)
    yv = yv.reshape((-1, 1))
    eigenmode = [1]
    wave_number = 16.0  # wave_number = freq/c
    waveguide_port_invar_numpy = {"x": np.zeros_like(yv), "y": yv}
    waveguide_port_outvar_numpy = {"Ez": 10 * eigvecs[:, 0:1]}

    # define geometry
    rec = Rectangle((0, 0), (width, height))
    # make list of nodes to unroll graph on
    maxwell = Maxwell2D(electric_field="Ez", magnetic_field=("Hx", "Hy"), k=wave_number * eps_sympy)
    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("Ez"), Key("Hx"), Key("Hy")],
        frequencies=("axis,diagonal", [i / 2.0 for i in range(int(wave_number * np.sqrt(eps1)) * 2 + 1)]),
        frequencies_params=("axis,diagonal", [i / 2.0 for i in range(int(wave_number * np.sqrt(eps1)) * 2 + 1)]),
        cfg=cfg.arch.modified_fourier,
    )
    nodes = maxwell.make_nodes() + [wave_net.make_node(name="wave_network")]

    waveguide_domain = Domain()

    PEC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"Ez": 0.0},
        batch_size=cfg.batch_size.PEC,
        lambda_weighting={"Ez": 100.0},
        criteria=Or(Eq(y, 0), Eq(y, height)),
    )

    waveguide_domain.add_constraint(PEC, "PEC")

    Waveguide_port = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar=waveguide_port_invar_numpy,
        outvar=waveguide_port_outvar_numpy,
        batch_size=cfg.batch_size.Waveguide_port,
        lambda_weighting={"Ez": np.full_like(yv, 0.5)},
    )
    waveguide_domain.add_constraint(Waveguide_port, "Waveguide_port")

    ABC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"normal_gradient_Hx": 0.0, "normal_gradient_Hy": 0.0},
        batch_size=cfg.batch_size.ABC,
        lambda_weighting={"normal_gradient_Hx": 10.0, "normal_gradient_Hy": 10.0},
        criteria=Eq(x, width),
    )
    waveguide_domain.add_constraint(ABC, "ABC")

    Interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"maxwell": 0.0},
        batch_size=cfg.batch_size.Interior,
        lambda_weighting={"maxwell": 1.0 / wave_number**2},
    )
    waveguide_domain.add_constraint(Interior, "Interior")

    # add inferencer data
    slab_inference = VoxelInferencer(
        bounds=[[0, 2], [0, 2]],
        npoints=[256, 256],
        nodes=nodes,
        output_names=["Ez", "Hx", "Hy"],
    )
    waveguide_domain.add_inferencer(slab_inference, "Inf" + str(int(wave_number)))

    # make solver
    slv = Solver(cfg, waveguide_domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()