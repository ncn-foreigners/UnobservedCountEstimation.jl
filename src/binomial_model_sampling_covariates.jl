


# TODO:: This is waaaaay to slow
# TODO:: This goes to infinity with sampling fix it!
# probably a math mistake in norm_const calc
function sample_M_matrix_variate_cond_random_eff(n, N, m, γ₀, γ₁, γ₂, u, M)
    # compute ξ, μ
    μ  = reduce(vcat, [(k .^ γ₁)' for k in eachrow(N)]) .* reduce(vcat, [(k .^ γ₂)' for k in eachrow(n ./ N)])
    μ  = 1 ./ (1 .+ 1 ./ μ)
    ξ₀ = vec(sum(N, dims = 2) .^ γ₀)
    ξ  = reduce(vcat, [(k .^ γ₁)' for k in eachrow(N)])
    

    norm_const = zeros(size(M)...)

    setprecision(100) do
        for i in axes(M, 1)
            for j in axes(M, 2)
                # add computable first term
                res = BigFloat(0)
                # compute minimum
                M_no_j = M[i, setdiff(axes(M, 2), j)]
                Wⱼ = minimum(M_no_j)
                if Wⱼ >= m[i, j]
                    for x in m[i, j]:Wⱼ
                        k = collect(0:x)
                        ress   = BigFloat.(k .* log(ξ₀[i]) .- k .* sum(log.(ξ[i, :])) .- logfactorial.(x .- k))
                        ress .-= [sum(logfactorial.(M_no_j .- t) .+ logfactorial(t)) for t in k]
                        ress   = sum(exp.(ress))
                        ress  *= exp(BigFloat(
                            x * log(ξ[i, j]) + x * log(1 - u[i] * μ[i, j]) + 
                            sum(logfactorial.(M_no_j)) + logfactorial(x) - logfactorial(x - m[i, j])
                        ))
                        res += ress
                    end # end for x
                end # end if
                # add second hyper geometric term
                for k in 0:Wⱼ
                    Tⱼ = max(Wⱼ + 1, m[i, j])
                    ress = exp(BigFloat(
                        Tⱼ * log(ξ[i, j] * (1 - u[i] * μ[i, j])) - logfactorial(Tⱼ - m[i, j]) - logfactorial(Tⱼ - k) +
                        k * log(ξ₀[i]) - k * sum(log.(ξ[i, :])) + sum(logfactorial.(M_no_j)) - 
                        sum(logfactorial.(M_no_j .- k)) - sum(logfactorial.(M_no_j * 0 .+ k))
                    ))
                    ress *= pFq((1, Tⱼ + 1), (Tⱼ - m[i, j] + 1, Tⱼ + 1 - k), ξ[i, j] * (1 - u[i] * μ[i, j]))
                    res  += ress
                end # end for k
                # copy needed?
                norm_const[i, j] = copy(res)
            end # end for j
        end # end for i
    end # end set precision

    # Conditional mass function at point x + m[i, j] for M[i,j]
    function mass_function(x, i, j)
        res = BigFloat(0)

        setprecision(100) do
            mm = min(minimum(M[i, setdiff(axes(M, 2), j)]), x + m[i, j])
            for z in 1:mm
                xxx  = sum(BigFloat.(logfactorial.(M[i, setdiff(axes(M, 2), j)]) - logfactorial.(M[i, setdiff(axes(M, 2), j)] .- z) .- logfactorial(z)))
                xxx += logfactorial(x + m[i,j]) - logfactorial(x + m[i, j] - z)
                xxx += z * log(ξ₀[i]) - z * sum(log.(ξ[i, :]))
                res += exp(xxx)
            end # end for

            res *= exp(BigFloat((x + m[i, j]) * log(ξ[i, j] * (1 - μ[i, j] * u[i])) - logfactorial(x)))
        end # end setprecision

        res
    end # end function

    ## Sampling
    #println(norm_const)
    #error("abc")
    U    =  rand(size(M)...)
    x    = zeros(Int, size(M)...)
    # not normalized CDF
    CDF  = reshape([mass_function(x[i, j], i, j) for j in axes(x, 2) for i in axes(x, 1)], size(x))
    # Multiplication is less prone to numerical errors
    U .*= norm_const
    cond = CDF .<= U

    # TODO:: Computationally this is the problem think about jumping by idk like 10 or even 100 since
    # on test code there is update of over 1300
    while any(cond)
        #println("--------")
        #println(sum(cond))
        #println(maximum(x))
        #println([minimum(U[cond] .- CDF[cond]), maximum(U[cond] .- CDF[cond])])
        x[cond] .+= 1

        # .+= or +=?
        CDF[cond] .+= reshape([cond[i,j] ? mass_function(x[i, j], i, j) : 0 for j in axes(x, 2) for i in axes(x, 1)], size(x))[cond]
        cond = CDF .< U
    end
    
    # return x + m
    x + m
end # end funciton