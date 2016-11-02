using Documenter, ImageQuilting

makedocs()

deploydocs(
  deps  = Deps.pip("mkdocs","python-markdown-math"),
  repo  = "github.com/juliohm/ImageQuilting.jl.git",
  julia = "0.5"
)
