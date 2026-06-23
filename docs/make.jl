using Documenter, ImageQuilting

makedocs(
  format=Documenter.HTML(prettyurls=get(ENV, "CI", "false") == "true"),
  sitename="ImageQuilting.jl",
  authors="Júlio Hoffimann",
  pages=[
    "Home" => "index.md",
    "Voxel reuse" => "voxelreuse.md",
    "About" => ["Author" => "about/author.md", "License" => "about/license.md", "Citation" => "about/citation.md"]
  ]
)

deploydocs(repo="github.com/JuliaEarth/ImageQuilting.jl.git")
