# ------------------------------------------------------------------
# Copyright (c) 2015, Júlio Hoffimann Mendes <juliohm@stanford.edu>
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

const HardData = Dict{NTuple{3,Integer},Real}
coords(hd::HardData) = keys(hd)
