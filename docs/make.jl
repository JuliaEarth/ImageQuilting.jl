# Workaround for JuliaLang/julia/pull/28625
if Base.HOME_PROJECT[] !== nothing
  Base.HOME_PROJECT[] = abspath(Base.HOME_PROJECT[])
end

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
  julia = "1.0"
)
