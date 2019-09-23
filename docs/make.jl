using Documenter, SparsityOperators

makedocs(
    modules = [SparsityOperators],
    format = :html,
    checkdocs = :exports,
    sitename = "SparsityOperators.jl",
    pages = Any["index.md"]
)

deploydocs(
    repo = "github.com/tknopp/SparsityOperators.jl.git",
)
