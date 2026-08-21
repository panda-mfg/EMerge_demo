"""128 MHz high-pass MRI birdcage coil.

The model uses 16 continuous rungs and 16 capacitor gaps in each end ring.
Two ports occupy half of two upper-ring gaps separated by 90 degrees; the
tuning capacitor occupies the other half, so the source and tuning element
are electrically in parallel without assigning two EMerge boundary
conditions to the same face.

Run a complete two-port sweep with::

    python examples/demo23_high_pass_birdcage.py

Generate and inspect only the mesh with::

    python examples/demo23_high_pass_birdcage.py --mesh-only --view

Use the optional NVIDIA cuDSS direct solver with::

    python examples/demo23_high_pass_birdcage.py --solver cudss

Capacitor tuning starts at 22 pF for this 16-rung, 240 mm diameter coil with
the saline phantom present. The EMerge/cuDSS sweep places the return-loss
minimum at 134 MHz with 22 pF and at 128 MHz with 24.1 pF, so 24.1 pF is the
validated default. Use ``--cap-pf 22`` to reproduce the starting sweep.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

import emerge as em
from emerge.plot import plot_sp, smith


mm = 1e-3
pF = 1e-12
MHz = 1e6
STARTING_CAPACITANCE_PF = 22.0
TUNED_CAPACITANCE_PF = 24.1


@dataclass(frozen=True)
class BirdcageParameters:
    target_frequency: float = 128 * MHz
    sweep_start: float = 118 * MHz
    sweep_stop: float = 138 * MHz
    sweep_points: int = 11

    rung_count: int = 16
    coil_radius: float = 120 * mm
    coil_length: float = 250 * mm
    strip_width: float = 10 * mm
    capacitor_gap: float = 5 * mm
    capacitance: float = TUNED_CAPACITANCE_PF * pF
    capacitor_q: float = 600.0
    feed_gap_indices: tuple[int, int] = (1, 5)

    phantom_radius: float = 90 * mm
    phantom_er: float = 78.0
    phantom_conductivity: float = 0.6
    phantom_density: float = 1000.0

    bore_radius: float = 320 * mm
    bore_length: float = 1600 * mm


@dataclass
class BirdcageGeometry:
    conductors: list
    capacitor_sheets: list
    port_sheets: list
    phantom: object
    bore: object


def annular_sector(
    inner_radius: float,
    outer_radius: float,
    angle_start: float,
    angle_stop: float,
    z: float,
    arc_sections: int = 1,
    name: str = "AnnularSector",
):
    """Create a planar annular sector at *z*.

    One section makes a straight-sided trapezoid, which is useful for a
    lumped sheet. Conductive ring arcs use several sections.
    """

    angles = np.linspace(angle_start, angle_stop, arc_sections + 1)
    x_outer = outer_radius * np.cos(angles)
    y_outer = outer_radius * np.sin(angles)
    x_inner = inner_radius * np.cos(angles[::-1])
    y_inner = inner_radius * np.sin(angles[::-1])

    polygon = em.geo.XYPolygon(
        np.r_[x_outer, x_inner],
        np.r_[y_outer, y_inner],
    )
    sector = polygon.geo(em.GCS.displace(0, 0, z), name=name)
    # XYPolygon.geo currently does not forward its name argument to the
    # resulting GeoPolygon, so preserve the descriptive model name here.
    sector.name = name
    return sector


def parallel_lossy_capacitor(capacitance: float, q: float, f_ref: float):
    """Return the impedance function for an ideal C in parallel with loss R."""

    resistance = q / (2 * np.pi * f_ref * capacitance)

    def impedance(frequency: float):
        admittance = 1 / resistance + 1j * 2 * np.pi * frequency * capacitance
        return 1 / admittance

    return impedance, resistance


def build_geometry(params: BirdcageParameters) -> BirdcageGeometry:
    pitch = 2 * np.pi / params.rung_count
    gap_angle = params.capacitor_gap / params.coil_radius
    radial_inner = params.coil_radius - params.strip_width / 2
    radial_outer = params.coil_radius + params.strip_width / 2
    z_lower = -params.coil_length / 2
    z_upper = params.coil_length / 2

    conductors = []
    capacitor_sheets = []
    port_sheets = []

    for rung_index in range(params.rung_count):
        rung_angle = -np.pi + rung_index * pitch
        radial = np.array([np.cos(rung_angle), np.sin(rung_angle), 0.0])
        tangent = np.array([-np.sin(rung_angle), np.cos(rung_angle), 0.0])

        rung_origin = (
            params.coil_radius * radial
            - params.strip_width / 2 * tangent
            + np.array([0.0, 0.0, z_lower])
        )
        rung = em.geo.Plate(
            rung_origin,
            params.strip_width * tangent,
            (0.0, 0.0, params.coil_length),
            name=f"Rung{rung_index + 1:02d}",
        ).set_material(em.lib.PEC)
        conductors.append(rung)

        # Each conductive arc is centered on a rung and stops at the two
        # adjacent capacitor gaps.
        arc_start = rung_angle - pitch / 2 + gap_angle / 2
        arc_stop = rung_angle + pitch / 2 - gap_angle / 2
        for ring_name, ring_z in (("Lower", z_lower), ("Upper", z_upper)):
            arc = annular_sector(
                radial_inner,
                radial_outer,
                arc_start,
                arc_stop,
                ring_z,
                arc_sections=8,
                name=f"{ring_name}RingArc{rung_index + 1:02d}",
            ).set_material(em.lib.PEC)
            conductors.append(arc)

        gap_center = rung_angle + pitch / 2
        gap_start = gap_center - gap_angle / 2
        gap_stop = gap_center + gap_angle / 2

        lower_cap = annular_sector(
            radial_inner,
            radial_outer,
            gap_start,
            gap_stop,
            z_lower,
            name=f"LowerCap{rung_index + 1:02d}",
        )
        capacitor_sheets.append(lower_cap)

        if rung_index in params.feed_gap_indices:
            # Split the radial strip into two side-by-side sheets. Both bridge
            # the same ring gap, giving the physical parallel C/feed topology
            # while keeping their boundary-condition tags distinct.
            upper_cap = annular_sector(
                radial_inner,
                params.coil_radius,
                gap_start,
                gap_stop,
                z_upper,
                name=f"UpperFeedCap{rung_index + 1:02d}",
            )
            port_sheet = annular_sector(
                params.coil_radius,
                radial_outer,
                gap_start,
                gap_stop,
                z_upper,
                name=f"UpperPort{len(port_sheets) + 1}",
            )
            capacitor_sheets.append(upper_cap)
            port_sheets.append(port_sheet)
        else:
            upper_cap = annular_sector(
                radial_inner,
                radial_outer,
                gap_start,
                gap_stop,
                z_upper,
                name=f"UpperCap{rung_index + 1:02d}",
            )
            capacitor_sheets.append(upper_cap)

    saline = em.Material(
        er=params.phantom_er,
        cond=params.phantom_conductivity,
        density=params.phantom_density,
        color="#4c8bd6",
        opacity=0.35,
        name="SalinePhantom",
    )
    phantom = em.geo.Sphere(params.phantom_radius).set_material(saline).foreground()

    bore_cs = em.GCS.displace(0, 0, -params.bore_length / 2)
    bore = (
        em.geo.Cylinder(
            params.bore_radius,
            params.bore_length,
            cs=bore_cs,
            Nsections=64,
            name="MRIBoreAir",
        )
        .set_material(em.lib.AIR)
        .background()
    )

    assert len(conductors) == 3 * params.rung_count
    assert len(capacitor_sheets) == 2 * params.rung_count
    assert len(port_sheets) == 2
    return BirdcageGeometry(
        conductors=conductors,
        capacitor_sheets=capacitor_sheets,
        port_sheets=port_sheets,
        phantom=phantom,
        bore=bore,
    )


def build_model(params: BirdcageParameters):
    model = em.Simulation("HighPassBirdcage128MHz")
    model.check_version("2.8.4")
    geometry = build_geometry(params)

    model.mw.set_frequency_range(
        params.sweep_start,
        params.sweep_stop,
        params.sweep_points,
    )
    model.mw.set_resolution(0.25)
    model.commit_geometry()

    # The small capacitor gaps control local accuracy. The saline mesh target
    # is about lambda/18 at 128 MHz for eps_r=78.
    model.mesher.set_face_size(em.select(*geometry.conductors), 8 * mm)
    model.mesher.set_face_size(em.select(*geometry.capacitor_sheets), 1.25 * mm)
    model.mesher.set_face_size(em.select(*geometry.port_sheets), 1.25 * mm)
    model.mesher.set_domain_size(geometry.phantom, 15 * mm)
    model.generate_mesh()

    capacitor_impedance, loss_resistance = parallel_lossy_capacitor(
        params.capacitance,
        params.capacitor_q,
        params.target_frequency,
    )
    for sheet in geometry.capacitor_sheets:
        radial_width = params.strip_width
        if "FeedCap" in sheet.name:
            radial_width /= 2
        mean_radius = (
            params.coil_radius - params.strip_width / 4
            if "FeedCap" in sheet.name
            else params.coil_radius
        )
        chord_length = (
            2 * mean_radius * np.sin(params.capacitor_gap / params.coil_radius / 2)
        )
        model.mw.bc.LumpedElement(
            sheet,
            impedance_function=capacitor_impedance,
            width=radial_width,
            height=chord_length,
        )

    for port_number, (gap_index, sheet) in enumerate(
        zip(params.feed_gap_indices, geometry.port_sheets), start=1
    ):
        gap_angle = -np.pi + (gap_index + 0.5) * 2 * np.pi / params.rung_count
        tangent = (-np.sin(gap_angle), np.cos(gap_angle), 0.0)
        mean_radius = params.coil_radius + params.strip_width / 4
        chord_length = (
            2 * mean_radius * np.sin(params.capacitor_gap / params.coil_radius / 2)
        )
        model.mw.bc.LumpedPort(
            sheet,
            port_number,
            width=params.strip_width / 2,
            height=chord_length,
            direction=tangent,
            Z0=50,
        )

    # The bore wall remains the default PEC boundary; only its axial ends are
    # opened. This mirrors an RF shield around the coil.
    model.mw.bc.AbsorbingBoundary(
        geometry.bore.front + geometry.bore.back,
        order=2,
        origin=(0.0, 0.0, 0.0),
    )

    return model, geometry, loss_resistance


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cap-pf",
        type=float,
        default=TUNED_CAPACITANCE_PF,
        help="capacitance in every end-ring gap (default: tuned 24.1 pF)",
    )
    parser.add_argument(
        "--mesh-only",
        action="store_true",
        help="stop after geometry, mesh, and boundary-condition setup",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="open the geometry/mesh viewer",
    )
    parser.add_argument(
        "--solver",
        choices=("auto", "cudss"),
        default="auto",
        help="linear solver backend (default: EMerge automatic selection)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    params = BirdcageParameters(capacitance=args.cap_pf * pF)
    model, geometry, loss_resistance = build_model(params)
    if args.solver == "cudss":
        model.set_solver(em.EMSolver.CUDSS)

    print(f"High-pass birdcage target: {params.target_frequency / MHz:.3f} MHz")
    print(f"Capacitance tuning start: {STARTING_CAPACITANCE_PF:.6g} pF")
    print(f"End-ring capacitors: {2 * params.rung_count} x {args.cap_pf:.6g} pF")
    print(f"Capacitor parallel-loss resistance: {loss_resistance / 1e3:.3f} kOhm")
    print(f"Feed gaps: {params.feed_gap_indices} (90 degrees apart)")
    print(f"Solver: {args.solver}")
    print(f"Mesh: {model.mesh.n_tets:,} tetrahedra")

    if args.view:
        model.view(
            selections=geometry.capacitor_sheets + geometry.port_sheets,
            plot_mesh=True,
            volume_mesh=False,
        )
        model.view(bc=True)

    if args.mesh_only:
        return

    data = model.mw.run_sweep()
    grid = data.scalar.grid
    dense_frequency = np.linspace(params.sweep_start, params.sweep_stop, 1001)
    s11 = grid.model_S(1, 1, dense_frequency)
    s22 = grid.model_S(2, 2, dense_frequency)
    s21 = grid.model_S(2, 1, dense_frequency)

    resonance_index = np.argmin(np.abs(s11))
    resonance = dense_frequency[resonance_index]
    print(f"Port-1 |S11| minimum: {resonance / MHz:.6f} MHz")

    plot_sp(
        dense_frequency,
        [s11, s22, s21],
        xunit="MHz",
        labels=["S11", "S22", "S21"],
    )
    smith(s11, f=dense_frequency, labels="S11")

    # Form the conventional circularly polarized drive: port 2 lags port 1
    # by 90 degrees. The normalization keeps total accepted drive comparable.
    field = data.field.find(freq=params.target_frequency)
    field.set_excitations(1 / np.sqrt(2), -1j / np.sqrt(2))
    model.display.add_objects(*geometry.conductors, geometry.phantom)
    model.display.animate().add_field(
        field.cutplane(5 * mm, z=0).scalar("normH", "complex"),
        symmetrize=False,
    )
    model.display.add_title("128 MHz quadrature |H| at isocenter")
    model.display.show()


if __name__ == "__main__":
    main()
