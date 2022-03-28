using Turing, Distributions, DifferentialEquations
using MCMCChains, Plots, StatsPlots
using Random
Random.seed!(14);
using PlotlySave
using LaTeXStrings


#### SIR
function SIR(du,u,p,t)
    θ = p
    S,I = u
    du[1] = -θ*S*I
    du[2] = θ*S*I - I
end

### Probemos con datos reales
# Cargamos datos de casos covid para la cdmx
using DataFrames, CSV
covid_cdmx = DataFrame(CSV.File("/home/milo/Documents/egap/clases/mod_sistemas/ode_gradiente/src/covid_cdmx.csv"))

odedata = collect(Array(covid_cdmx)')

# Condiciones iniciales
S0 = 1 - covid_cdmx.contagiados[1]
I0 = covid_cdmx.contagiados[1]

u0 = [S0,I0]
p = 3

tspan = (1.0,58.0+(2/7))
prob1 = ODEProblem(SIR,u0,tspan,p)
sol1 = solve(prob1,Euler(),dt = 1/7)

Turing.setadbackend(:forwarddiff)

@model function fitSIR(data, prob1)
    # Priors, P(θ)
    θ ~ truncated(Normal(1.5,2),0,10)
    # Resolvemos el sistema ODE, modelo muestral P(y|θ)
    p = θ
    prob = remake(prob1, p=p)
    predicted = solve(prob,Euler(),dt = 1/7)

    for i = 1:length(predicted)
        data[:,i] ~ MvNormal(predicted[i], [0.2 0; 0 0.3])
    end
end

model = fitSIR(odedata, prob1)

# Corremos 3 cadenas de forma independiente sin usar multithreading.
chain = mapreduce(c -> sample(model, NUTS(.8),2500), chainscat, 1:5)
chain_resultados = plot(chain)

pl = StatsPlots.scatter(1:402, vcat(odedata'...)[403:804]);
chain_array = Array(chain)
for k in 1:300
    resol = solve(remake(prob1,p=chain_array[rand(1:1500)]),Euler(),dt = 1/7)
    plot!(vcat(resol.u'...)[403:804], alpha=0.1, color = "#BBBBBB", legend = false)
end

###########################################################
###########################################################
###########################################################

lockdown_times = [1.0, 20.0]
condition(u,t,integrator) = t ∈ lockdown_times
function affect!(integrator)
    if integrator.t < lockdown_times[2]
        integrator.p = 1
    else
        integrator.p = 3
    end
end
cb = PresetTimeCallback(lockdown_times, affect!);
prob1 = ODEProblem(SIR,u0,tspan,p)
sol1 = solve(prob1,Euler(),dt = 1/7, callback = cb)

Turing.setadbackend(:forwarddiff)

@model function fitSIR(data, prob1)
    # Priors, P(θ)
    θ ~ truncated(Normal(1.5,2),0,10)
    # Resolvemos el sistema ODE, modelo muestral P(y|θ)
    p = θ
    prob = remake(prob1, p=p)
    predicted = solve(prob,Euler(),dt = 1/7, callback = cb)

    for i = 1:length(predicted)
        data[:,i] ~ MvNormal(predicted[i], [0.2 0; 0 0.3])
    end
end

model = fitSIR(odedata, prob1)

# Corremos 3 cadenas de forma independiente sin usar multithreading.
chain = mapreduce(c -> sample(model, NUTS(.6),200), chainscat, 1:4)
chain_resultados = plot(chain)

pl = StatsPlots.scatter(1:402, vcat(odedata'...)[403:804]);
chain_array = Array(chain)
for k in 1:300
    resol = solve(remake(prob1,p=chain_array[rand(1:200)]),Euler(),dt = 1/7)
    plot!(vcat(resol.u'...)[403:804], alpha=0.1, color = "#BBBBBB", legend = false)
end
