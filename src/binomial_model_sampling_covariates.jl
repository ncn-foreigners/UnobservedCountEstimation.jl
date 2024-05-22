function sample_M_matrix_variate_cond_random_eff(n, N, m, γ₀, γ₁, γ₂, u, M)
    # compute ξ, μ
    μ  = reduce(vcat, [(k .^ γ₁)' for k in eachrow(N)]) .* reduce(vcat, [(k .^ γ₂)' for k in eachrow(n ./ N)])
    μ  = 1 ./ (1 .+ 1 ./ μ)
    ξ₀ = vec(sum(N, dims = 2) .^ γ₀)
    ξ  = reduce(vcat, [(k .^ γ₁)' for k in eachrow(N)])
    

    norm_const = zeros(size(M)...)

    for i in axes(M, 1)
        for j in axes(M, 2)
            # add computable first term
            res = BigFloat(0)
            # compute minimum
            Wⱼ = minimum(M[i, setdiff(axes(M, 2), j)])
            if Wⱼ >= m[i, j]
                for x in m[i, j]:Wⱼ
                    k = collect(0:x)
                    ress   = k * log(ξ₀[i]) - k * sum(log.(ξ[i, :])) - logfactorial.(x .- k)
                    ress .-= [sum(logfactorial.(M[i, setdiff(axes(M, 2), j)] .- t)) for t in k]
                    ress   = sum(ress)
                    # this is a scalar
                    ress  *= exp(BigFloat(
                        x * log(ξ[i, j]) + x * log(1 - u[i] * μ[i, j])   + 
                        sum(logfactorial.(M[i, setdiff(axes(M, 2), j)])) + 
                        logfactorial(x) - logfactorial(x - m[i, j])
                    ))
                    res += ress
                end # end for x
            end # end if
            # add second hyper geometric term
            for k in 0:Wⱼ
                Tⱼ = max(Wⱼ, m[i, j])
                ress = exp(BigFloat(
                    (Tⱼ + 1) * log(ξ[i, j] * (1 - u[i] * μ[i, j])) - logfactorial(Tⱼ + 1 - m[i, j]) - logfactorial(Tⱼ + 1 - k) +
                    log(ξ₀[i]) - sum(log.(ξ[i, :])) + sum(log.(M[i, setdiff(axes(M, 2), j)])) - sum(log.(M[i, setdiff(axes(M, 2), j)] .- k))
                ))
                ress *= pFq((1, Tⱼ + 2), (Tⱼ - m[i, j] + 2, Tⱼ + 2 - k), ξ[i, j] * (1 - u[i] * μ[i, j]))
                res  += ress
            end # end for k
            # copy needed?
            norm_const[i, j] = copy(res)
        end # end for j
    end # end for i

    # Conditional mass function at point x + m[i, j] for M[i,j]
    function mass_function(x, i, j)
        mm = min(minimum(M[i, setdiff(axes(M, 2), j)]), x)
        res = BigFloat(0)

        for z in 1:mm
            xxx  = sum(logfactorial.(M[i, setdiff(axes(M, 2), j)]) - logfactorial.(M[i, setdiff(axes(M, 2), j)] .- z))
            xxx += z * log.(ξ₀[i]) - z * sum(log.(ξ[i, :]))
            res += exp(BigFloat(xxx))
        end # end for

        # compute normalization constant
        res *= exp(BigFloat(x * log(ξ[i, j] * (1 - μ[i, j] * u[i])) - logfactorial(x) - norm_const[i, j]))
    end # end function

    ## Sampling
    U    =  rand(size(M)...)
    x    = zeros(Int, size(M)...)
    xxx  = reshape([mass_function(x[i, j], i, j) for j in axes(x, 2) for i in axes(x, 1)], size(x))
    cond = xxx .<= U

    while any(cond)
        x[cond] .+= 1

        xxx += reshape([mass_function(x[i, j], i, j) for j in axes(x, 2) for i in axes(x, 1)], size(x))
        cond = xxx .< U
    end
    
    # return x + m
    x + m
end # end funciton