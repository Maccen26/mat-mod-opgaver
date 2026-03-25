using Printf
using JuMP, HiGHS

function load_heights(path)
    values = Float64[]
    for raw in readlines(path)
        s = strip(raw)
        if isempty(s)
            continue
        end
        push!(values, parse(Float64, s))
    end
    return values
end

H = load_heights(joinpath(@__DIR__, "heights.txt"))


# Define K
K = [300.0, 140.0, 40.0]

function solveIP(H, K)
    h = length(H)
    CHD = 10.0

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    set_attribute(model, "time_limit", 10.0)
    set_attribute(model, "presolve", "on")
    set_attribute(model, "mip_rel_gap", 0.0)

    @variable(model, x[1:h], Bin)
    @variable(model, R[1:h] >= 0)
    
    
    # Objective: minimize number of nukes
    # @objective(model, Min, sum(x)) # Experiment 1

    # Adjacency constraint: no two adjacent positions can both have a nuke
    #@constraint(model, [i in 1:h-1], x[i] + x[i+1] <= 1) # Experiment 2

    # Objective: minimize sum of residuals
    @objective(model, Min, sum((R[i] - H[i] - CHD) for i in 1:h)) # Experiment 2

    # Coverage constraints
    for i in 1:h
        l = max(1, i - 2)
        r = min(h, i + 2)
        
        # Minimum coverage requirement
        @constraint(model, R[i] >= H[i] + CHD)
        
        # Coverage calculations
        @constraint(model, R[i] == sum(K[1] * x[j] for j in l:r if j == i) +
                                sum(K[2] * x[j] for j in l:r if abs(j - i) == 1) +
                                sum(K[3] * x[j] for j in l:r if abs(j - i) == 2))
    end

    solve_time = @elapsed optimize!(model)

    status = termination_status(model)
    if status != MOI.OPTIMAL
        println("Optimize was not successful. Return code: ", status)
        println("Solve time (s): ", round(solve_time, digits=4))
        return
    end

    xval = Int.(round.(value.(x)))
    println("Objective value: ", Int(round(objective_value(model))))
    println("Solve time (s): ", round(solve_time, digits=4))
    println("Station placements (1 for station, 0 for no station): ", xval)
end

solveIP(H, K)

