using Documenter, ImageQuilting

makedocs(
  format = :html,
  sitename = "ImageQuilting.jl",
  authors = "Júlio Hoffimann Mendes",
  pages = [
    "Home" => "index.md",
    "Concepts" => "concepts.md",
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

deploydocs(
  repo  = "github.com/juliohm/ImageQuilting.jl.git",
  target = "build",
  deps = nothing,
  make = nothing,
  julia = "0.5"
)
