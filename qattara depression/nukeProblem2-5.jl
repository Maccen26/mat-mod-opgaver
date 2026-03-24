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


# Define K
K = [300.0, 140.0, 40.0]

# function constructA(H,K)
#     h = length(H)
#     A = spzeros(h,h)
#     for i in 1:h
#         for j in 1:h
#             if i == j
#                 A[i,j] = K[1]
#             elseif abs(i-j) == 1
#                 A[i,j] = K[2]
#             elseif abs(i-j) == 2
#                 A[i,j] = K[3]
#             else
#                 A[i,j] = 0.0
#             end
#         end
#     end
#     return A
# end

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
    
    
    # Adjacency constraint: no two adjacent positions can both have a station
    @constraint(model, [i in 1:h-1], x[i] + x[i+1] <= 1) # Problem 5

    # Objective: minimize sum of residuals
    # @objective(model, Min, sum(x)) # Problem 3 (should not be in the other problems)
    @objective(model, Min, sum((R[i] - H[i] - CHD) for i in 1:h)) # Problem 4

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

