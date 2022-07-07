using Turing, Distributions, DifferentialEquations
using MCMCChains, Plots, StatsPlots
using Random
Random.seed!(14);
using PlotlySave
using LaTeXStrings
using LinearAlgebra

# ODE
function bacteria(du,u,p,t)
    α,β = p
    N = u[1]
    du[1] = α*N*(1-β*N)
end

# Condiciones iniciales
N0 = 3.0
u0 = [N0]
p = [0.4,0.04]
init_t = 0.0
final_t = 10.0
tiempo = (init_t,final_t)

prob1 = ODEProblem(bacteria,u0,tiempo,p)
sol = solve(prob1,Euler(),dt=0.1)

model_data = vcat(sol.u...)
experiment_data = [rand(Normal(i,1)) for i in model_data]
t = 0.0:0.1:10
#plot(experiment_data,seriestype = :scatter)

bacterias_experimento = plot(t,experiment_data,seriestype = :scatter, title = "Datos del experimento",xlabel = "Tiempo, días",ylabel = "Conteos,m", label ="Experimentos")
#save("bacterias_experimento.pdf", bacterias_experimento)

bacterias_modelo = plot(t,model_data, title = "Datos del modelo",xlabel = "Tiempo, días",ylabel = "Conteos,m", label = L"N(t)",color=:red,legend=:bottomright)
#save("bacterias_modelo.pdf", bacterias_modelo)
#save("bacterias_modelo_experimentos.pdf", bacterias_experimento)
#plot!(t,model_data)

### Reescribimos el modelo de crecimiento de bacterias para resolver el problema de estimación de parámetros con Turing
#Turing.setadbackend(:forwarddiff)

@model function fitbacteria(data, prob1)
    # Priors, P(θ)
    α ~ Uniform(0,2)
    β ~ Uniform(0,1)
    # Resolvemos el sistema ODE, modelo muestral P(y|θ)
    p = [α,β]
    prob = remake(prob1, p=p)
    predicted = solve(prob,Tsit5(),saveat=0.1)

    for i = 1:length(predicted)
        data[i] ~ Normal(predicted.u[i][1], 0.3)
    end
end

odedata = Array(sol) + 0.8 * randn(size(Array(sol)))

model = fitbacteria(experiment_data, prob1)

# Corremos 5 cadenas de forma independiente sin usar multithreading.
chain = mapreduce(c -> sample(model, NUTS(.65),2500), chainscat, 1:5)
chain_resultados = plot(chain)
#save("chain_resultados.pdf", chain_resultados)

# Recolección de datos
pl = StatsPlots.scatter(sol.t, odedata', label = "Experimento");

chain_array = Array(chain)
for k in 1:2000
    resol = solve(remake(prob1,p=chain_array[rand(1:1500), 1:2]),Tsit5(),saveat=0.1)
    plot!(resol, alpha=0.1, legend = false,color=:yellow)
    if k ==2000
        plot!(resol, alpha=0.1,color=:yellow,label = "Inferido")
    end
end
# display(pl)
plot!(sol, w=1,color=:red, label = L"N(t)")
#save("final_plot.pdf", pl)



###########################################################
#### MODELO SI
###########################################################
#### SI

function SI(du,u,p,t)
    θ = p
    S,I = u
    du[1] = -θ*S*I
    du[2] = θ*S*I - I
end
# Condiciones iniciales
S0 = 0.99
I0 = 0.01

u0 = [S0,I0]
p = 3
#t = 0.0:0.01:7.0
#tspan = (0.0,10.0)
t = 1.0:1/48:10.0
tspan = (1.0,10.0)
prob1 = ODEProblem(SI,u0,tspan,p)
sol = DifferentialEquations.solve(prob1,Euler(),dt = 1/7)

plot(sol)


### Reescribimos el modelo de crecimiento de bacterias para resolver el problema de estimación de parámetros con Turing
#Turing.setadbackend(:forwarddiff)

@model function fitSI(data, prob1)
    # Priors, P(θ)
    θ ~ Uniform(0,10)
    # Resolvemos el sistema ODE, modelo muestral P(y|θ)
    p = θ
    prob = remake(prob1, p=p)
    predicted = solve(prob,Tsit5(),saveat=0.1)

    for i = 1:length(predicted)
        data[:,i] ~ MvNormal(predicted[i], [0.2 0; 0 0.3])
    end
end

sol1 = solve(prob1,Tsit5(),saveat=0.1)
odedata = Array(sol1) + 0.04 * randn(size(Array(sol1)))

# Valor real vs Perturbado
plot(sol1; alpha=0.3)
scatter!(sol1.t, odedata'; color=[1 2], label="")

model = fitSI(odedata, prob1)

# Corremos 3 cadenas de forma independiente sin usar multithreading.
chain = mapreduce(c -> sample(model, NUTS(.8),2500), chainscat, 1:5)
chain_resultados = plot(chain)

pl = StatsPlots.scatter(sol1.t, odedata');
chain_array = Array(chain)
for k in 1:300
    resol = solve(remake(prob1,p=chain_array[rand(1:1500)]),Tsit5(),saveat=0.1)
    plot!(resol, alpha=0.1, color = "#BBBBBB", legend = false)
end
plot!(sol1, w=1, legend = false)

###########################################################
#### Modelo PneumonicPlague
###########################################################

function PneumonicPlague(du,u,p,t)

    infection_rate,contact_rate,decisive_time,antibiotics_coverage = p

    susceptible_population,infected_population,recovering_population,deceased_population = u
    total_population = 10000

    #auxiliary variables
    infected_fraction  =  infected_population/total_population
    if antibiotics_coverage == 1.0
        fatality_ratio = 0.15
    elseif antibiotics_coverage == 0.0
        fatality_ratio = 0.9
    else
        fatality_ratio = 0
    end


    #flow variables
      #leer la redacción + diagrama
    infections = susceptible_population*infection_rate*contact_rate*infected_fraction
    recovering = ((1-fatality_ratio)*infected_population)/decisive_time
    #1=los que se van a recuperar
    #fatality_ratio=los que morirán
    deaths = (fatality_ratio*infected_population)/decisive_time

    #state variables
    du[1] =  (-1)*infections
    du[2] =  infections-recovering-deaths
    du[3] =  recovering
    du[4] =  deaths

end


# Condiciones iniciales
susceptible_population = 9999
infected_population = 1
recovering_population = 0
deceased_population = 0

u0 = [susceptible_population,infected_population,
      recovering_population,deceased_population]

# Parámetros
infection_rate = 0.75
contact_rate = 50/7
decisive_time = 2
antibiotics_coverage = 1

p = [infection_rate,contact_rate,decisive_time,antibiotics_coverage]

tspan = (0.0,30.0)
prob1 = ODEProblem(PneumonicPlague,u0,tspan,p)
sol = DifferentialEquations.solve(prob1,RK4(),dt = 1)
plot(sol)


### Reescribimos el modelo PneumonicPlague para resolver el problema de estimación de parámetros con Turing
#Turing.setadbackend(:forwarddiff)

@model function fitPneumonicPlague(data, prob1)
    # Priors, P(θ)
    infection_rate ~ Uniform(0,10)
    contact_rate ~ Uniform(0,10)
    decisive_time ~ Uniform(0,10)
    antibiotics_coverage ~ Uniform(0,10)
    # Resolvemos el sistema ODE, modelo muestral P(y|θ)
    p = infection_rate,contact_rate,decisive_time,antibiotics_coverage
    prob = remake(prob1, p=p)
    predicted = solve(prob,Tsit5(),saveat=0.1)

    for i = 1:length(predicted)
        data[:,i] ~ MvNormal(predicted[i], 0.3 * I)
    end
end

sol1 = solve(prob1,Tsit5(),saveat=0.1)
odedata = Array(sol1) + 400 * randn(size(Array(sol1)))

# Valor real vs Perturbado
plot(sol1; alpha=0.3)
scatter!(sol1.t, odedata'; color=[1 2 3 4], label="")


model = fitPneumonicPlague(odedata, prob1)

# Corremos 3 cadenas de forma independiente sin usar multithreading.
chain = mapreduce(c -> sample(model, NUTS(.8),2000), chainscat, 1)
chain_resultados = plot(chain)

pl = StatsPlots.scatter(sol1.t, odedata');
chain_array = Array(chain)
for k in 1:300
    resol = solve(remake(prob1,p=chain_array[rand(1:1500),:]),Tsit5(),saveat=0.1)
    plot!(resol, alpha=0.1, color = "#BBBBBB", legend = false)
end
plot!(sol1, w=1, legend = false)


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
prob1 = ODEProblem(SI,u0,tspan,p)
sol1 = solve(prob1,Euler(),dt = 1/7)

Turing.setadbackend(:forwarddiff)

@model function fitSI(data, prob1)
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

model = fitSI(odedata, prob1)

# Corremos 3 cadenas de forma independiente sin usar multithreading.
chain = mapreduce(c -> sample(model, NUTS(.8),2500), chainscat, 1:5)
chain_resultados = plot(chain)

pl = StatsPlots.scatter(1:402, vcat(odedata'...)[403:804]);
chain_array = Array(chain)
for k in 1:300
    resol = solve(remake(prob1,p=chain_array[rand(1:1500)]),Euler(),dt = 1/7)
    plot!(vcat(resol.u'...)[403:804], alpha=0.1, color = "#BBBBBB", legend = false)
end
