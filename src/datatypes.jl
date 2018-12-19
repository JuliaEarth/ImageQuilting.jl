# ------------------------------------------------------------------
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

const HardData = Dict{CartesianIndex{3},Real}
coords(hd::HardData) = keys(hd)
