# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

"""
    IQ(var₁=>param₁, var₂=>param₂, ...)

Image quilting simulation solver as described in Hoffimann et al. 2017.

## Parameters

### Required

* `trainimg` - Training image from which to extract tiles
* `tilesize` - Tuple with tile size for each dimension

### Optional

* `overlap`  - Overlap size (default to (1/6, 1/6, ..., 1/6))
* `path`     - Simulation path (`:raster` (default), `:dilation`, or `:random`)
* `mapping`  - Data mapping method (default to `NearestMapping()`)
* `inactive` - Vector of inactive voxels (i.e. `CartesianIndex`) in the grid
* `soft`     - A pair `(data,dataTI)` of geospatial data objects (default to `nothing`)
* `tol`      - Initial relaxation tolerance in (0,1] (default to `0.1`)

## Global parameters

### Optional

* `threads`      - Number of threads in FFT (default to number of physical CPU cores)
* `showprogress` - Whether to show or not the estimated time duration (default to false)
* `rng`          - Random number generator (default to `Random.GLOBAL_RNG`)

## References

* Hoffimann et al 2017. *Stochastic simulation by image quilting of process-based geological models.*
* Hoffimann et al 2015. *Geostatistical modeling of evolving landscapes by means of image quilting.*
"""
@simsolver IQ begin
  @param trainimg
  @param tilesize
  @param overlap       = nothing
  @param path          = :raster
  @param mapping       = NearestMapping()
  @param inactive      = nothing
  @param soft          = nothing
  @param tol           = 0.1
  @global threads      = cpucores()
  @global showprogress = false
  @global rng          = Random.GLOBAL_RNG
end

function preprocess(problem::SimulationProblem, solver::IQ)
  # retrieve problem info
  pdata   = data(problem)
  pdomain = domain(problem)
  simsize = size(pdomain)
  Dim     = embeddim(pdomain)

  # result of preprocessing
  preproc = Dict{Symbol,Tuple}()

  for covars in covariables(problem, solver)
    for var in covars.names
      # get user parameters
      varparams = covars.params[(var,)]

      # training image as simple array
      TI = varparams.trainimg
      trainimg = asarray(TI, var)

      # default overlap
      overlap = isnothing(varparams.overlap) ? ntuple(i->1/6, Dim) :
                                               varparams.overlap

      # determine data mappings
      vmapping = if hasdata(problem)
        map(pdata, pdomain, (var,), varparams.mapping)[var]
      else
        Dict()
      end

      # create soft data object
      soft = if !isnothing(varparams.soft)
        data, dataTI = varparams.soft
        @assert domain(data) == pdomain "incompatible soft data for target domain"
        @assert domain(dataTI) == domain(TI) "incompatible soft data for training image"
        schema   = Tables.schema(values(data))
        schemaTI = Tables.schema(values(dataTI))
        vars     = schema.names |> collect |> sort
        varsTI   = schemaTI.names |> collect |> sort
        @assert vars == varsTI "variables for target domain and training image differ"
        [(asarray(data, var), asarray(dataTI, var)) for var in vars]
      else
        []
      end

      # create hard data object
      hdata = Dict{CartesianIndex{Dim},Real}()
      for (loc, datloc) in vmapping
        push!(hdata, lin2cart(simsize, loc) => pdata[var][datloc])
      end

      # disable inactive voxels
      shape = Dict{CartesianIndex{Dim},Real}()
      if !isnothing(varparams.inactive)
        for icoords in varparams.inactive
          push!(shape, icoords => NaN)
        end
      end

      hard = merge(hdata, shape)

      preproc[var] = (varparams, trainimg, simsize, overlap, soft, hard)
    end
  end

  preproc
end

function solvesingle(::SimulationProblem, covars::NamedTuple, solver::IQ, preproc)
  # random number generator
  rng = solver.rng

  varreal = map(covars.names) do var
    # unpack preprocessed parameters
    par, trainimg, simsize, overlap, soft, hard = preproc[var]

    # run image quilting core function
    reals = iqsim(trainimg, par.tilesize, simsize;
                  overlap=overlap, path=par.path,
                  soft=soft, hard=hard,
                  tol=par.tol, threads=solver.threads,
                  showprogress=solver.showprogress, rng=rng)

    # flatten result
    var => vec(reals[1])
  end

  Dict(varreal)
end
