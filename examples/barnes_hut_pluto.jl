### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 1931b2d2-aced-416c-bf62-a302d70a3c71
begin
	using Pkg
	Pkg.activate(".")
end

# ╔═╡ aa000003-1111-4111-8111-000000000003
begin
    using NearestNeighbors
    using StaticArrays
    using WGLMakie
    using PlutoUI
    using Random
    using LinearAlgebra: norm, dot
    import AbstractTrees: children
end

# ╔═╡ aa000001-1111-4111-8111-000000000001
md"""
# Barnes–Hut galaxy collider 🌌

An interactive N-body simulation built on the **NearestNeighbors.jl tree-walking API**.
A `KDTree` is rebuilt every step and drives the whole Barnes–Hut algorithm:

* `postorder(tree)` — children-before-parent traversal, used to aggregate node masses and centers of mass
* `children(node)` — descending during the force walk
* `treeregion(node)` — node bounding boxes, for the opening criterion *and* for drawing the tree
* `leafpoints(node)` / `leaf_point_indices(node)` — direct summation over the bodies in a leaf

Rendering uses **WGLMakie**, which draws with WebGL right inside the notebook and pushes
`Observable` updates to the browser — much faster than re-rasterizing with Cairo.
(If the figure ever shows up blank, run `WGLMakie.Page()` in a new cell and re-run the figure cell.)
"""

# ╔═╡ aa000004-1111-4111-8111-000000000004
md"""
## Gravity via the tree-walking API
"""

# ╔═╡ aa000005-1111-4111-8111-000000000005
begin
    const Vec2 = SVector{2,Float64}
    G::Float64 = 1.0
    DT::Float64 = 0.002    # leapfrog time step
    LEAFSIZE::Int = 24 # KDTree leaf size
end;

# ╔═╡ aa000006-1111-4111-8111-000000000006
begin
    # Build the KDTree and compute, for every node, its total mass and center of
    # mass. `postorder` guarantees children are visited before their parent, so an
    # internal node can combine the already-computed values of its two children.
    function build_gravity_tree(positions, masses; leafsize, parallel = false)
        tree = KDTree(positions; leafsize, parallel)
        n_nodes = length(postorder(tree))
        node_mass = zeros(n_nodes)
        node_com = fill(zero(eltype(positions)), n_nodes)
        for node in postorder(tree)
            i = node.index
            cs = children(node)
            if isempty(cs) # leaf: accumulate the stored bodies
                m = 0.0
                c = zero(eltype(positions))
                for (p, j) in zip(leafpoints(node), leaf_point_indices(node))
                    m += masses[j]
                    c += masses[j] * p
                end
                node_mass[i] = m
                node_com[i] = c / m
            else
                left, right = cs
                m = node_mass[left.index] + node_mass[right.index]
                node_mass[i] = m
                node_com[i] = (node_mass[left.index] * node_com[left.index] +
                               node_mass[right.index] * node_com[right.index]) / m
            end
        end
        return tree, node_mass, node_com
    end

    # Gravitational acceleration at point `p`, walking down from `node`.
    # A node — leaf or internal — is accepted as a single pseudo-particle when it
    # is far away relative to its size: s/d < θ, with s the widest side of the
    # node's bounding box (from `treeregion`) and d the distance to its center of
    # mass. An internal node that fails the test is opened; a leaf that fails it
    # is summed body by body (the body at `p` itself contributes exactly zero
    # because Δ = 0 there).
    function acceleration(node, node_mass, node_com, masses, p, G, θ², ε²)
        i = node.index
        δ = node_com[i] - p
        d² = dot(δ, δ)
        region = treeregion(node)
        s = maximum(region.maxes - region.mins)
        if s^2 < θ² * d² # far away: treat the whole node as one body at its COM
            r² = d² + ε²
            return (G * node_mass[i] / (r² * sqrt(r²))) * δ
        end
        cs = children(node)
        if isempty(cs) # nearby leaf: direct summation
            a = zero(p)
            for (q, j) in zip(leafpoints(node), leaf_point_indices(node))
                Δ = q - p
                r² = dot(Δ, Δ) + ε²
                a += (G * masses[j] / (r² * sqrt(r²))) * Δ
            end
            return a
        end
        left, right = cs
        return acceleration(left, node_mass, node_com, masses, p, G, θ², ε²) +
               acceleration(right, node_mass, node_com, masses, p, G, θ², ε²)
    end
end

# ╔═╡ aa000007-1111-4111-8111-000000000007
md"""
## Initial conditions and time stepping

Two rotating disks, each bound by a heavy central body, on a slightly-bound prograde
flyby with periapsis ≈ 7 (computed from two-body orbital mechanics, so it stays a
graze for any companion mass). The center of mass is at rest at the origin.
"""

# ╔═╡ aa000008-1111-4111-8111-000000000008
begin
    mutable struct SimState
        pos::Vector{Vec2}
        vel::Vector{Vec2}
        acc::Vector{Vec2}
        masses::Vector{Float64}
        t::Float64
    end

    function add_galaxy!(rng, pos, vel, masses; n, center, vcenter, radius,
                         central_mass, disk_mass, spin)
        push!(pos, Vec2(center))
        push!(vel, Vec2(vcenter))
        push!(masses, central_mass)
        m_star = disk_mass / n
        for _ in 1:n
            r = 0.6 + radius * rand(rng)^2 # concentrated toward the core
            ϕ = 2π * rand(rng)
            dir = Vec2(cos(ϕ), sin(ϕ))
            push!(pos, Vec2(center) + r * dir)
            # circular orbit around the (dominant) central body + a little dispersion
            v = sqrt(G * (central_mass + disk_mass * min(r / radius, 1.0)) / r)
            tangent = spin * Vec2(-dir[2], dir[1])
            push!(vel, Vec2(vcenter) + v * tangent + 0.03v * randn(rng, Vec2))
            push!(masses, m_star)
        end
    end

    # `θ` and `ε` are only used for the accelerations of the very first
    # half-kick and deliberately default here instead of coming from the
    # sliders: wiring them to the sliders would make the state cell depend on
    # them, restarting the simulation on every drag.
    function make_initial_state(; n_main, n_comp, comp_mass_frac, seed = 42,
                                θ = 0.6, ε = 0.35)
        rng = Xoshiro(seed)
        pos = Vec2[]
        vel = Vec2[]
        masses = Float64[]

        main_central, main_disk, main_radius = 1200.0, 240.0, 11.0
        comp_central = comp_mass_frac * main_central
        comp_disk = comp_mass_frac * main_disk
        comp_radius = max(3.0, main_radius * sqrt(comp_mass_frac))

        M1 = main_central + main_disk
        M2 = comp_central + comp_disk
        μ = G * (M1 + M2)

        # Two-body orbit of the galaxy centers: slightly bound, periapsis ≈ 7.
        d = Vec2(28.0, 10.0)               # initial separation
        r0 = norm(d)
        v = 0.86 * sqrt(2μ / r0)           # just below escape speed
        a_orbit = -μ / (v^2 - 2μ / r0)     # semi-major axis from the vis-viva equation
        rp = 7.0
        e = 1 - rp / a_orbit
        L = sqrt(μ * rp * (1 + e))         # orbital angular momentum
        r̂ = d / r0
        t̂ = Vec2(-r̂[2], r̂[1])              # prograde w.r.t. the disks' spin
        v_t = L / r0
        v_r = -sqrt(max(v^2 - v_t^2, 0.0)) # approaching
        v_rel = v_r * r̂ + v_t * t̂

        # center-of-mass frame
        c1 = -(M2 / (M1 + M2)) * d
        c2 = d + c1
        v1 = -(M2 / (M1 + M2)) * v_rel
        v2 = v_rel + v1

        add_galaxy!(rng, pos, vel, masses; n = n_main, center = c1, vcenter = v1,
                    radius = main_radius, central_mass = main_central,
                    disk_mass = main_disk, spin = +1)
        add_galaxy!(rng, pos, vel, masses; n = n_comp, center = c2, vcenter = v2,
                    radius = comp_radius, central_mass = comp_central,
                    disk_mass = comp_disk, spin = +1)

        acc = fill(zero(Vec2), length(pos))
        tree, node_mass, node_com = build_gravity_tree(pos, masses; leafsize = LEAFSIZE)
        root = treeroot(tree)
        for i in eachindex(pos)
            acc[i] = acceleration(root, node_mass, node_com, masses, pos[i], G, θ^2, ε^2)
        end
        return SimState(pos, vel, acc, masses, 0.0)
    end

    # Second half-kick for a single body: walk the tree for its acceleration,
    # update its velocity and store the acceleration for the next step's first
    # kick. Bodies are independent (read-only tree, disjoint writes), so this
    # loop body is safe to run in parallel over `i`.
    function halfkick!(state, i, root, node_mass, node_com, dt, G, θ², ε²)
        a = acceleration(root, node_mass, node_com, state.masses,
                         state.pos[i], G, θ², ε²)
        state.vel[i] += 0.5dt * a
        state.acc[i] = a
    end

    # `nsubsteps` leapfrog (kick-drift-kick) steps. Returns the last tree so the
    # caller can draw it, plus the seconds spent building trees and computing
    # forces (summed over the substeps). With `threaded = true` the tree is built
    # in parallel and the force loop is spread over `Threads.nthreads()` threads.
    function advance!(state::SimState, nsubsteps; dt, G, θ, ε, leafsize, threaded)
        local tree
        t_build = 0.0
        t_force = 0.0
        θ², ε² = θ^2, ε^2
        for _ in 1:nsubsteps
            @. state.vel += 0.5dt * state.acc
            @. state.pos += dt * state.vel
            t0 = time_ns()
            tree, node_mass, node_com =
                build_gravity_tree(state.pos, state.masses; leafsize, parallel = threaded)
            t1 = time_ns()
            root = treeroot(tree)
            if threaded
                Threads.@threads for i in eachindex(state.pos)
                    halfkick!(state, i, root, node_mass, node_com, dt, G, θ², ε²)
                end
            else
                for i in eachindex(state.pos)
                    halfkick!(state, i, root, node_mass, node_com, dt, G, θ², ε²)
                end
            end
            t_build += (t1 - t0) / 1e9
            t_force += (time_ns() - t1) / 1e9
            state.t += dt
        end
        return tree, t_build, t_force
    end
end

# ╔═╡ aa000009-1111-4111-8111-000000000009
# Collect the bounding box of every tree node as line segments, fading the
# shallow (large) cells and brightening the deep (small) ones.
function tree_cell_segments!(segs, cols, tree)
    empty!(segs)
    empty!(cols)
    function visit(node, depth)
        region = treeregion(node)
        x0, y0 = region.mins
        x1, y1 = region.maxes
        c = RGBAf(0.35, 0.85, 1.0, min(0.03 + 0.045 * depth, 0.4))
        append!(segs, (Point2f(x0, y0), Point2f(x1, y0),
                       Point2f(x1, y0), Point2f(x1, y1),
                       Point2f(x1, y1), Point2f(x0, y1),
                       Point2f(x0, y1), Point2f(x0, y0)))
        for _ in 1:8
            push!(cols, c)
        end
        for child in children(node)
            visit(child, depth + 1)
        end
    end
    visit(treeroot(tree), 0)
    return segs, cols
end

# ╔═╡ aa00000a-1111-4111-8111-00000000000a
md"""
## Setup — changing these (or pressing restart) rebuilds the galaxies

Main-galaxy stars: $(@bind n_main Slider(200:100:3000, default = 900, show_value = true))

Companion stars: $(@bind n_comp Slider(50:50:1000, default = 300, show_value = true))

Companion mass (fraction of main): $(@bind comp_mass_frac Slider(0.05:0.05:1.0, default = 0.25, show_value = true))
"""

# ╔═╡ aa00000b-1111-4111-8111-00000000000b
@bind restart Button("💥 Restart simulation")

# ╔═╡ aa00000c-1111-4111-8111-00000000000c
md"""
## Live parameters — take effect immediately

Opening angle θ: $(@bind θ Slider(0.0:0.05:1.5, default = 0.6, show_value = true))
*(θ = 0 is exact summation; larger θ is faster and coarser)*

Softening ε: $(@bind ε Slider(0.05:0.05:1.0, default = 0.35, show_value = true))

Steps per frame: $(@bind substeps Slider(1:30, default = 12, show_value = true))

Target fps: $(@bind target_fps Slider([30, 45, 60, 90, 120, 240], default = 60, show_value = true))

Show tree cells: $(@bind show_tree CheckBox(default = true))

Threaded (build tree + forces on all threads): $(@bind threaded CheckBox(default = Threads.nthreads() > 1))
*(only helps when Julia is started with `-t auto`/`JULIA_NUM_THREADS`; currently $(Threads.nthreads()) thread(s) available)*
"""

# ╔═╡ aa00000d-1111-4111-8111-00000000000d
md"""
▶ Run simulation: $(@bind running CheckBox(default = true))
"""

# ╔═╡ aa00000e-1111-4111-8111-00000000000e
state = begin
    restart # pressing the button re-creates the initial conditions
    make_initial_state(; n_main, n_comp, comp_mass_frac)
end;

# ╔═╡ aa00000f-1111-4111-8111-00000000000f
begin
    core_mask = state.masses .> 10 # the two heavy central bodies
    star_mask = .!core_mask

    star_pts = Observable(Point2f.(state.pos[star_mask]))
    star_speed = Observable(Float32.(norm.(state.vel[star_mask])))
    core_pts = Observable(Point2f.(state.pos[core_mask]))
    cell_segs = Observable(Point2f[])
    cell_cols = Observable(RGBAf[])

    space_cmap = cgrad([RGBf(0.15, 0.25, 0.7), RGBf(0.45, 0.65, 1.0),
                        RGBf(0.95, 0.95, 0.9), RGBf(1.0, 0.8, 0.45)])

    fig = Figure(size = (900, 560), backgroundcolor = :black)
    ax = Axis(fig[1, 1], backgroundcolor = :black, aspect = DataAspect(),
              limits = (-30, 30, -18.75, 18.75))
    hidedecorations!(ax)
    hidespines!(ax)

    linesegments!(ax, cell_segs; color = cell_cols, linewidth = 0.6)
    # stars, drawn twice: a soft halo plus a bright core
    scatter!(ax, star_pts; color = star_speed, colormap = space_cmap,
             colorrange = (0, 22), markersize = 7, alpha = 0.15)
    scatter!(ax, star_pts; color = star_speed, colormap = space_cmap,
             colorrange = (0, 22), markersize = 2)
    scatter!(ax, core_pts; color = RGBAf(1, 1, 0.9, 0.25), markersize = 22)
    scatter!(ax, core_pts; color = RGBf(1, 1, 0.95), markersize = 6)

    info_label = Observable("")
    # space = :relative pins the label to the axis viewport, so it stays in the
    # corner when panning/zooming
    text!(ax, 0.012, 0.02; text = info_label, space = :relative,
          color = RGBAf(0.7, 0.8, 0.9, 0.55), fontsize = 13,
          align = (:left, :bottom))

    fig
end

# ╔═╡ aa000012-1111-4111-8111-000000000012
# Persistent state for the animation loop. This cell has no dependencies, so it
# survives re-runs of the loop cell below:
#   * `loop_gen` — generation counter used to stop stale loop tasks
#   * `fps_last`/`fps_ema` — frame-rate tracker (exponential moving average)
begin
    loop_gen = Ref(0)
    fps_last = Ref(UInt64(0))
    fps_ema = Ref(0.0)
end;

# ╔═╡ aa000010-1111-4111-8111-000000000010
# Animation loop. Frames are produced by a background task that updates the
# Makie Observables directly — WGLMakie pushes them to the browser over its own
# (Bonito) websocket, so a frame never pays the Pluto reactive round trip
# (which caps a Clock-driven loop at ~30 fps).
#
# Pluto reactivity is still what handles *parameter changes*: this cell depends
# on `state`, `running` and the live sliders, so any change re-runs it. The
# re-run bumps `loop_gen`; the previous task sees the stale generation at the
# top of its next iteration and exits, and a fresh task starts with the new
# parameter values.
begin
    mygen = (loop_gen[] += 1)
    fps_last[] = 0 # fresh fps measurement for the new loop
    if running
        @async try
            while loop_gen[] == mygen
                frame_start = time_ns()
                tree, t_build, t_force =
                    advance!(state, substeps; dt = DT, G, θ, ε, leafsize = LEAFSIZE,
                             threaded)
                if show_tree
                    tree_cell_segments!(cell_segs[], cell_cols[], tree)
                else
                    empty!(cell_segs[])
                    empty!(cell_cols[])
                end
                notify(cell_segs)
                notify(cell_cols)
                star_pts[] = Point2f.(state.pos[star_mask])
                star_speed[] = Float32.(norm.(state.vel[star_mask]))
                core_pts[] = Point2f.(state.pos[core_mask])

                if fps_last[] != 0
                    inst = 1e9 / (frame_start - fps_last[])
                    fps_ema[] = fps_ema[] == 0 ? inst : 0.9 * fps_ema[] + 0.1 * inst
                end
                fps_last[] = frame_start
                fps_text = fps_ema[] > 0 ? " · $(round(fps_ema[], digits = 1)) fps" : ""
                mode = threaded ? "$(Threads.nthreads()) threads" : "serial"
                info_label[] = "$(length(state.pos)) bodies · θ = $θ · $mode · t = $(round(state.t, digits = 1))\n" *
                               "tree $(round(1000t_build, digits = 1)) ms · " *
                               "forces $(round(1000t_force, digits = 1)) ms$fps_text"

                # Frame pacing. Always yield once so other tasks (websocket
                # sender, Pluto) get scheduled even when we're over budget.
                # `sleep` has millisecond granularity and consistently wakes
                # *late* (a plain sleep-the-remainder here caps at ~53 fps),
                # so sleep only the coarse part of the budget and spin out the
                # last couple of milliseconds with `yield`.
                yield()
                budget = 1 / target_fps
                elapsed() = (time_ns() - frame_start) / 1e9
                if budget - elapsed() > 0.004
                    sleep(budget - elapsed() - 0.002)
                end
                while elapsed() < budget
                    yield()
                end
            end
        catch err
            info_label[] = "animation loop died: " * sprint(showerror, err)
        end
    end
    loop_status = running ? "running" : "paused"
    # Markdown.parse on an already-interpolated string, because the md""
    # macro parses markdown *before* interpolating and garbles values that
    # sit next to `#` or `**` markers.
    Markdown.parse("animation loop #$mygen — **$loop_status**")
end

# ╔═╡ Cell order:
# ╟─aa000001-1111-4111-8111-000000000001
# ╠═1931b2d2-aced-416c-bf62-a302d70a3c71
# ╠═aa000003-1111-4111-8111-000000000003
# ╟─aa000004-1111-4111-8111-000000000004
# ╠═aa000005-1111-4111-8111-000000000005
# ╠═aa000006-1111-4111-8111-000000000006
# ╟─aa000007-1111-4111-8111-000000000007
# ╠═aa000008-1111-4111-8111-000000000008
# ╠═aa000009-1111-4111-8111-000000000009
# ╟─aa00000a-1111-4111-8111-00000000000a
# ╟─aa00000b-1111-4111-8111-00000000000b
# ╟─aa00000c-1111-4111-8111-00000000000c
# ╟─aa00000d-1111-4111-8111-00000000000d
# ╟─aa00000e-1111-4111-8111-00000000000e
# ╟─aa00000f-1111-4111-8111-00000000000f
# ╟─aa000012-1111-4111-8111-000000000012
# ╟─aa000010-1111-4111-8111-000000000010
