# Workaround for GR warnings
ENV["GKSwstype"] = "100"

using Documenter, ImageQuilting

istravis = "TRAVIS" ∈ keys(ENV)

makedocs(
  format = Documenter.HTML(prettyurls=istravis),
  sitename = "ImageQuilting.jl",
  authors = "Júlio Hoffimann",
  pages = [
    "Home" => "index.md",
    "Examples" => "examples.md",
    "Voxel reuse" => "voxel-reuse.md",
    "GPU support" => "gpu-support.md",
    "About" => [
      "Author" => "about/author.md",
      "License" => "about/license.md",
      "Citation" => "about/citation.md"
    ]
  ]
)

deploydocs(repo="github.com/JuliaEarth/ImageQuilting.jl.git")
