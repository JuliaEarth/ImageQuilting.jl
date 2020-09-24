# ------------------------------------------------------------------
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

"""
    ImgQuilt(var₁=>param₁, var₂=>param₂, ...)

Image quilting simulation solver as described in Hoffimann et al. 2017.

## Parameters

### Required

* `TI`       - Training image from which to extract tiles
* `tilesize` - Tuple with tile size for each dimension

### Optional

* `overlap`  - Overlap size (default to (1/6, 1/6, ..., 1/6))
* `path`     - Simulation path (:raster (default), :dilation, or :random)
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
  @param overlap       = nothing
  @param path          = :raster
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
  dims = ncoords(pdomain)
  simsize = size(pdomain)

  # sanity checks
  @assert pdomain isa RegularGrid "ImgQuilt requires RegularGrid domain"

  # result of preprocessing
  preproc = Dict{Symbol,Tuple}()

  for covars in covariables(problem, solver)
    for var in covars.names
      # get user parameters
      varparams = covars.params[(var,)]

      # default overlap
      overlap = varparams.overlap ≠ nothing ? varparams.overlap :
                                              ntuple(i->1/6, dims)

      # create hard data object
      hdata = Dict{CartesianIndex{dims},Real}()
      for (loc, datloc) in datamap(problem, var)
        push!(hdata, lin2cart(simsize, loc) => pdata[var][datloc])
      end

      # disable inactive voxels
      shape = Dict{CartesianIndex{dims},Real}()
      if varparams.inactive ≠ nothing
        for icoords in varparams.inactive
          push!(shape, icoords => NaN)
        end
      end

      preproc[var] = (varparams, simsize, overlap, merge(hdata, shape))
    end
  end

  preproc
end

function solvesingle(problem::SimulationProblem, covars::NamedTuple,
                     solver::ImgQuilt, preproc)
  varreal = map(covars.names) do var
    # unpack preprocessed parameters
    par, simsize, overlap, hard = preproc[var]

    # run image quilting core function
    reals = iqsim(par.TI, par.tilesize, simsize;
                  overlap=overlap, path=par.path,
                  soft=par.soft, hard=hard, tol=par.tol,
                  threads=solver.threads, gpu=solver.gpu,
                  showprogress=solver.showprogress)

    # flatten result
    var => vec(reals[1])
  end

  Dict(varreal)
end
