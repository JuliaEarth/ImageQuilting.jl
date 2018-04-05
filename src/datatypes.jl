# ------------------------------------------------------------------
# Copyright (c) 2015, Júlio Hoffimann Mendes <juliohm@stanford.edu>
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

struct SoftData
  function SoftData(A, B)
    Base.depwarn("`SoftData(AUX, TI -> AUXTI)` is deprecated, please pass `soft=[(AUX,AUXTI)]` to `iqsim` instead.", :SoftData)
  end
end

const HardData = Dict{NTuple{3,Integer},Real}
coords(hd::HardData) = keys(hd)
