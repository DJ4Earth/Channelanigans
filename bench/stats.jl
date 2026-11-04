# Shared sample statistics (included by bench/common.jl, run_cases.jl, make_table.jl,
# export_raw.jl) so every summary uses the same definition:
#   mean, sample standard deviation (n-1), and a two-sided 95% Student-t confidence
#   interval for the mean: mean ± t_{0.975, n-1} · s / √n.

if !@isdefined(STATS_JL_LOADED)
const STATS_JL_LOADED = true

# Two-sided 95% Student-t critical values t_{0.975, df}, df = 1…30 (normal beyond).
const T975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
              2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
              2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)
t975(df) = df <= length(T975) ? T975[df] : 1.960

"Mean, sample standard deviation and 95% Student-t confidence interval of `x`."
function sample_stats(x)
    x = collect(Float64, x)
    n = length(x)
    n == 0 && return nothing
    m = sum(x) / n
    s = n > 1 ? sqrt(sum(abs2, x .- m) / (n - 1)) : NaN
    h = n > 1 ? t975(n - 1) * s / sqrt(n) : NaN
    sorted = sort(x)
    med = isodd(n) ? sorted[(n + 1) ÷ 2] : (sorted[n ÷ 2] + sorted[n ÷ 2 + 1]) / 2
    return Dict{String, Any}("n" => n, "mean" => m, "std" => s, "ci95_halfwidth" => h,
                "ci95" => [m - h, m + h], "median" => med,
                "min" => sorted[1], "max" => sorted[end], "values" => x)
end

end # if
