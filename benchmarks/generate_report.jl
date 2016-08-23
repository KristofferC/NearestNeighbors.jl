# The functionality to export the benchmark results to a markdown file heavily uses code from Nanosoldier.jl
# The Nanosoldier.jl package is licensed under the MIT "Expat" License:
#
# Copyright (c) 2015: Jarrett Revels.

const REGRESS_MARK = ":x:"
const IMPROVE_MARK = ":white_check_mark:"

function printreport(io::IO, results; iscomparisonjob::Bool = false)

    if iscomparisonjob
        print(io, """
                  A ratio greater than `1.0` denotes a possible regression (marked with $(REGRESS_MARK)), while a ratio less
                  than `1.0` denotes a possible improvement (marked with $(IMPROVE_MARK)). Only significant results - results
                  that indicate possible regressions or improvements - are shown below (thus, an empty table means that all
                  benchmark results remained invariant between builds).

                  | ID | time ratio | memory ratio |
                  |----|------------|--------------|
                  """)
    else
        print(io, """
                  | ID | time | GC time | memory | allocations |
                  |----|------|---------|--------|-------------|
                  """)
    end

    entries = BenchmarkTools.leaves(results)

    try
        entries = entries[sortperm(map(x -> string(first(x)), entries))]
    end


    for (ids, t) in entries
        if !(iscomparisonjob) || BenchmarkTools.isregression(t) || BenchmarkTools.isimprovement(t)
            println(io, resultrow(ids, t))
        end
    end

    return nothing
end

idrepr(id) = (str = repr(id); str[searchindex(str, '['):end])
intpercent(p) = string(ceil(Int, p * 100), "%")
resultrow(ids, t::BenchmarkTools.Trial) = resultrow(ids, minimum(t))

function resultrow(ids, t::BenchmarkTools.TrialEstimate)
    t_tol = intpercent(BenchmarkTools.params(t).time_tolerance)
    m_tol = intpercent(BenchmarkTools.params(t).memory_tolerance)
    timestr = string(BenchmarkTools.prettytime(BenchmarkTools.time(t)), " (", t_tol, ")")
    memstr = string(BenchmarkTools.prettymemory(BenchmarkTools.memory(t)), " (", m_tol, ")")
    gcstr = BenchmarkTools.prettytime(BenchmarkTools.gctime(t))
    allocstr = string(BenchmarkTools.allocs(t))
    return "| `$(idrepr(ids))` | $(timestr) | $(gcstr) | $(memstr) | $(allocstr) |"
end

function resultrow(ids, t::BenchmarkTools.TrialJudgement)
    t_tol = intpercent(BenchmarkTools.params(t).time_tolerance)
    m_tol = intpercent(BenchmarkTools.params(t).memory_tolerance)
    t_ratio = @sprintf("%.2f", BenchmarkTools.time(BenchmarkTools.ratio(t)))
    m_ratio =  @sprintf("%.2f", BenchmarkTools.memory(BenchmarkTools.ratio(t)))
    t_mark = resultmark(BenchmarkTools.time(t))
    m_mark = resultmark(BenchmarkTools.memory(t))
    timestr = "$(t_ratio) ($(t_tol)) $(t_mark)"
    memstr = "$(m_ratio) ($(m_tol)) $(m_mark)"
    return "| `$(idrepr(ids))` | $(timestr) | $(memstr) |"
end

resultmark(sym::Symbol) = sym == :regression ? REGRESS_MARK : (sym == :improvement ? IMPROVE_MARK : "")
