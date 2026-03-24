using Printf
using JuMP, HiGHS

function load_heights(path)
    values = Float64[]
    for raw in readlines(path)
        s = strip(raw)
        if isempty(s) || s == "[" || s == "]"
            continue
        end
        push!(values, parse(Float64, s))
    end
    return values
end

H = load_heights(joinpath(@__DIR__, "heights.txt"))


# Define three K settings
K_settings = [
    [300.0, 140.0, 40.0],   # Setting 1
    [500.0, 230.0, 60.0],   # Setting 2
    [1000.0, 400.0, 70.0]   # Setting 3
]
num_settings = length(K_settings)

function solveIP(H, K_settings)
    h = length(H)
    CHD = 10.0

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    set_attribute(model, "time_limit", 60.0)
    set_attribute(model, "presolve", "on")
    set_attribute(model, "mip_rel_gap", 0.00) 

    # x[i,k] = 1 if position i uses setting k, 0 otherwise
    @variable(model, x[1:h, 1:num_settings], Bin)
    @variable(model, R[1:h] >= 0)
    
    # At most one setting per position
    @constraint(model, [i in 1:h], sum(x[i, k] for k in 1:num_settings) <= 1)
    
    # Adjacency constraint: no two adjacent positions can both have a station (regardless of setting)
    @constraint(model, [i in 1:h-1], 
        sum(x[i, k] for k in 1:num_settings) + sum(x[i+1, k] for k in 1:num_settings) <= 1)

    # Objective: minimize sum of residuals
    @objective(model, Min, sum((R[i] - H[i] - CHD) for i in 1:h))

    # Coverage constraints
    for i in 1:h
        l = max(1, i - 2)
        r = min(h, i + 2)
        
        # Minimum coverage requirement
        @constraint(model, R[i] >= H[i] + CHD)
        
        # Coverage formula: R[i] = sum over neighbors j of (K_setting[k] for setting k of position j) * x[j,k]
        # This requires understanding which setting each j uses
        coverage_expr = AffExpr()
        for j in l:r
            for k in 1:num_settings
                K = K_settings[k] # Get the K values for the current setting
                dist = abs(i - j) # Distance from position i to neighbor j
                coeff = dist == 0 ? K[1] : dist == 1 ? K[2] : K[3] # Coefficient based on distance and setting
                add_to_expression!(coverage_expr, coeff, x[j, k]) # Add coeff * x[j,k] to the coverage expression
            end
        end
        @constraint(model, R[i] == coverage_expr)
    end

    solve_time = @elapsed optimize!(model)

    status = termination_status(model)
    println("Termination status: ", status)
    println("Solve time (s): ", round(solve_time, digits=4))
    
    if status == MOI.OPTIMAL
        println("Found optimal solution.")
    else
        println("Optimize was not successful.")
        return
    end

    if !has_values(model)
        return
    end

    xval = Int.(round.(value.(x)))
    println("Objective value: ", Int(round(objective_value(model))))
    println("Solve time (s): ", round(solve_time, digits=4))
    
    # Display which setting each position uses
    println("\nStation configuration:")
    for i in 1:h
        for k in 1:num_settings
            if xval[i, k] == 1
                println(i, " ", k)
            end
        end
    end
end

solveIP(H, K_settings)