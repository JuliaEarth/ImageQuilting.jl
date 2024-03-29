using Documenter, ImageQuilting

isCI = "CI" ∈ keys(ENV)

makedocs(
  format=Documenter.HTML(prettyurls=isCI),
  sitename="ImageQuilting.jl",
  authors="Júlio Hoffimann",
  pages=[
    "Home" => "index.md",
    "Voxel reuse" => "voxel-reuse.md",
    "About" => ["Author" => "about/author.md", "License" => "about/license.md", "Citation" => "about/citation.md"]
  ]
)

deploydocs(repo="github.com/JuliaEarth/ImageQuilting.jl.git")
