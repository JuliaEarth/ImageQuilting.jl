## Copyright (c) 2014 Júlio Hoffimann Mendes
##
## This file is part of ImageQuilting.
##
## ImageQuilting is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## ImageQuilting is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with ImageQuilting.  If not, see <http://www.gnu.org/licenses/>.
##
## Created: 13 Aug 2014
## Author: Júlio Hoffimann Mendes

using Images

include("imquilt.jl")

function example()
    srand(2014) # make sure results are reproducible
    imagesdir = joinpath(Pkg.dir(),"ImageQuilting","src","images")
    imquilt(imread(joinpath(imagesdir,"radishes.jpg")), 64, 5 , show=true)
    imquilt(imread(joinpath(imagesdir,"weave.jpg"))   , 80, 5 , show=true)
    imquilt(imread(joinpath(imagesdir,"brick.jpg"))   , 80, 5 , show=true)
    imquilt(imread(joinpath(imagesdir,"apples.gif"))  , 31, 15, show=true)
    imquilt(imread(joinpath(imagesdir,"btile.tif"))   , 16, 15, show=true)
end
