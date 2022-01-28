### A Pluto.jl notebook ###
# v0.17.5

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 35a26f9e-c6b2-40c3-baa4-b01077f1b551
begin
	using Pkg
	Pkg.add("PlutoUI")
	Pkg.add("DifferentialEquations")
	Pkg.add("Plots")
end

# ╔═╡ 9229714d-6084-4985-a3f3-b1495fedbc36
using PlutoUI,DifferentialEquations, Plots

# ╔═╡ 95e952af-b57a-442f-9d70-fb0dc3822c0c
begin

function growth_collapse(du,u,p,t)

	# initial parameters
normal_birth_rate,regeneration_rate,carrying_capacity,minimum_regeneration_rate,rapid_resource_depletion_time,renewable_resource_consumption_per_capita,normal_lifetime,percent_consumers = p

	# initial conditions
  	population , renewable_resources = u 
	
	#auxiliary endogenous variables
	consumer_population = population*percent_consumers
	per_capita_renewable_resource_availability = renewable_resources/population
	resource_availability_dependent_lifetime = max(15,min(100,
	                                                         normal_lifetime*
	                                                         per_capita_renewable_resource_availability))
	minimum_regeneration = carrying_capacity*minimum_regeneration_rate
	resource_dependent_regeneration = regeneration_rate*
	                                 renewable_resources*
	                                 (renewable_resources/carrying_capacity)*
	                                 (1-(renewable_resources/carrying_capacity))
	#flow variables
	 births_flow = population*per_capita_renewable_resource_availability*normal_birth_rate
	 deaths_flow = consumer_population/resource_availability_dependent_lifetime
	 regeneration = minimum_regeneration+resource_dependent_regeneration
	 resource_use = min(population*renewable_resource_consumption_per_capita,
	                   renewable_resources/rapid_resource_depletion_time)
	#state variables
	 dpopulation = births_flow-deaths_flow
	 drenewable_resources = regeneration-resource_use
	
	 du .=(dpopulation,drenewable_resources)
end
struct TwoColumn{L, R}
    left::L
    right::R
end

function Base.show(io, mime::MIME"text/html", tc::TwoColumn)
    write(io, """<div style="display: flex;"><div style="flex: 50%;">""")
    show(io, mime, tc.left)
    write(io, """</div><div style="flex: 50%;">""")
    show(io, mime, tc.right)
    write(io, """</div></div>""")
end
end

# ╔═╡ d0de4fa5-7858-43bc-bedb-1080c9512cd6
begin 
	normal_birth_rate_slider = @bind normal_birth_rate Slider(0.002:0.0001:0.009, show_value=true,default = 0.35/100)
	regeneration_rate_slider = @bind regeneration_rate Slider(0.8:0.005:1.5,show_value=true,default = 1.2)
	carrying_capacity_slider = @bind carrying_capacity Slider(6e6:1000:8e6,show_value=true,default = 7.5e6)
	minimum_regeneration_rate_slider = @bind minimum_regeneration_rate Slider(0.001:0.001:
0.019,show_value=true, default = 1/100)
	rapid_resource_depletion_time_slider = @bind rapid_resource_depletion_time Slider(0.1:0.001:1.9,show_value=true,default = 1)
	renewable_resource_consumption_per_capita_slider = @bind renewable_resource_consumption_per_capita Slider(0.1:0.001:1.9,show_value=true, default = 1)
	normal_lifetime_slider = @bind normal_lifetime Slider(10:1:50,show_value=true,default = 30)
	percent_consumers_slider = @bind percent_consumers Slider(0.1:0.001:1.5,show_value=true, default = 0.90)
	
	population_slider = @bind population Slider(0.5e6:0.001e6:2e6, show_value=true, default = 1e6)
	renewable_resources_slider = @bind renewable_resources Slider(4e6:0.001e6:6e6,show_value=true, default = 5e6)

TwoColumn(md"""**Parameters**


	Normal birth rate: $(normal_birth_rate_slider)
	
	Regeneration rate: $(regeneration_rate_slider)

	Carrying capacity: $(carrying_capacity_slider)

	Minimum regeneration rate : $(minimum_regeneration_rate_slider)

	Rapid resource depletion time : $(rapid_resource_depletion_time_slider)

	Renewable resource consumption per capita : $(renewable_resource_consumption_per_capita_slider)

	Normal lifetime : $(normal_lifetime_slider)

	Percent consumers : $(percent_consumers_slider)""", md"""**Initial Conditions**
	
	Population: $(population_slider)
	
	Renewable resources: $(renewable_resources_slider)""")


end

# ╔═╡ 34814736-b09a-4bb1-890e-321815e135df
begin
	# Save de initial state variable in an array
	u0= [population,renewable_resources]
	
	# Save parameters in an array 
	dt = 0.1    # 1 Quarterly
	D = 100.0 # Simulate for 30 years
	
	p =[normal_birth_rate,
		regeneration_rate,
		carrying_capacity,
		minimum_regeneration_rate,
		rapid_resource_depletion_time,
		renewable_resource_consumption_per_capita,
		normal_lifetime,percent_consumers]

	# Solve the ODE system
	prob1 = ODEProblem(growth_collapse,u0,(0.0,D),p)
	sol = DifferentialEquations.solve(prob1,RK4();dt=0.1,adaptive=false)

	# Plot state variables
	plot(sol, title = "State variables dynamics", label = ["Population" "Renewable resources"], lw = 3)

end

# ╔═╡ Cell order:
# ╠═35a26f9e-c6b2-40c3-baa4-b01077f1b551
# ╠═9229714d-6084-4985-a3f3-b1495fedbc36
# ╟─95e952af-b57a-442f-9d70-fb0dc3822c0c
# ╟─d0de4fa5-7858-43bc-bedb-1080c9512cd6
# ╟─34814736-b09a-4bb1-890e-321815e135df
