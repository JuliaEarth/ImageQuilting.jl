using PkgBenchmark

function ask_version()
  print("What version would you like to compare to? ")
  chomp(strip(readline(STDIN)))
end

results = judge("ImageQuilting", ask_version())

showall(PkgBenchmark.benchmarkgroup(results))

println()
