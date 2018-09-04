# ------------------------------------------------------------------
# Copyright (c) 2017, Júlio Hoffimann Mendes <juliohm@stanford.edu>
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

"""
    ImgQuilt(var₁=>param₁, var₂=>param₂, ...)

Image quilting simulation solver as described in Hoffimann et al. 2017.

## Parameters

### Required

* `TI`       - Training image
* `tilesize` - Tile size in x, y and z

### Optional

* `overlap`  - Overlap size in x, y and z (default to (1/6, 1/6, 1/6))
* `cut`      - Boundary cut algorithm (:boykov (default) or :dijkstra)
* `path`     - Simulation path (:rasterup (default), :rasterdown, :dilation, or :random)
* `simplex`  - Whether to apply or not the simplex transform (default to false)
* `inactive` - Vector of inactive voxels (i.e. tuples (i,j,k)) in the grid
* `soft`     - A vector of `(data,dataTI)` pairs
* `tol`      - Initial relaxation tolerance in (0,1] (default to 0.1)

## Global parameters

### Optional

* `threads`      - Number of threads in FFT (default to number of physical CPU cores)
* `gpu`          - Whether to use the GPU or the CPU (default to false)
* `showprogress` - Whether to show or not the estimated time duration (default to false)
"""
@simsolver ImgQuilt begin
  @param TI
  @param tilesize
  @param overlap       = (1/6, 1/6, 1/6)
  @param cut           = :boykov
  @param path          = :rasterup
  @param simplex       = false
  @param inactive      = nothing
  @param soft          = []
  @param tol           = .1
  @global threads      = cpucores()
  @global gpu          = false
  @global showprogress = false
end

function preprocess(problem::SimulationProblem, solver::ImgQuilt)
  # retrieve problem info
  pdata = data(problem)
  pdomain = domain(problem)

  # sanity checks
  @assert pdomain isa RegularGrid "ImgQuilt solver only supports regular grids"
  @assert ndims(pdomain) ∈ [2,3] "Number of dimensions must be 2 or 3"

  # result of preprocessing
  preproc = Dict{Symbol,Tuple}()

  for (var, V) in variables(problem)
    @assert var ∈ keys(solver.params) "Parameters for variable `$var` not found"

    # get user parameters
    varparams = solver.params[var]

    # add ghost dimension to simulation grid if necessary
    gridsize = ndims(pdomain) == 2 ? (size(pdomain)..., 1) : size(pdomain)

    # create hard data object
    hdata = HardData()
    for (loc, datloc) in datamap(problem, var)
      push!(hdata, myind2sub(gridsize, loc) => value(pdata, datloc, var))
    end

    # disable inactive voxels
    shape = HardData()
    if varparams.inactive ≠ nothing
      for icoords in varparams.inactive
        push!(shape, icoords => NaN)
      end
    end

    preproc[var] = (varparams, gridsize, merge(hdata, shape))
  end

  preproc
end

function solve_single(problem::SimulationProblem, var::Symbol,
                      solver::ImgQuilt, preproc)
  # unpack preprocessed parameters
  par, gridsize, hard = preproc[var]

  # run image quilting core function
  reals = iqsim(par.TI, par.tilesize, gridsize;
                overlap=par.overlap,
                soft=par.soft, hard=hard, tol=par.tol,
                cut=par.cut, path=par.path,
                threads=solver.threads, gpu=solver.gpu,
                showprogress=solver.showprogress)

  # flatten result
  reals[1][:]
end
