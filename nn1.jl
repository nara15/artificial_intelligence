
type RedNeuronal

	entradasTamanio::Array{Int64, 1}
	usa_bias::Bool
	ritmo_aprendizaje::Float64
	momentum::Float64
	
	nodosEntrada::Array{Float64, 1}
	nodosOcultos::Array{Float64, 1}
	nodosSalida::Array{Float64, 1}
	
	wi::Array{Float64, 2}					# wi - Pesos de entrada
	wo::Array{Float64, 2}					# wo - Pesos de salida
	
	pesos_ci::Array{Float64, 2}				# ci
	pesos_co::Array{Float64, 2}				# co
	
	funcion_calc_pesos_iniciales::Function
	funcion_propagacion::Function
	derivada_funcion_propagacion::Function

end


function RedNeuronal(p_estructura::Array{Int64, 1}, p_usa_bias::Bool)
	
	dim_red = p_estructura
	dim_pesos = length(p_estructura) - 1
	
	RedNeuronal(p_estructura,							#	Tamaño de la red neuronal
				p_usa_bias,								#	Indica si usa bias
				0.25,									#	Ritmo de aprendizaje
				0.1,									#	Momentum
				ones(dim_red[1]),						#	Neuronas de entrada
				ones(dim_red[2]),						#	Neuronas de la capa oculta
				ones(dim_red[3]),						#	Neuronas de la capa de salida
				rand(dim_red[1], dim_red[2]),			#	Pesos de entrada
				rand(dim_red[2], dim_red[3]),			#	Pesos de salida
				rand(dim_red[1], dim_red[2]),			#	Pesos desde la capa de entrada hasta ocultos
				rand(dim_red[2], dim_red[3]),			#	Pesos deltas
				() -> rand(-3.0:3.0),
				(x::Float64) -> 1/(1+exp(-1*(x))),
				(y::Float64) -> y*(1-y)
				)			
end

#	Inicializar las redes neuronales ===================================
function ini_red(p_estructura_red::Array{Int64,1})
	red = RedNeuronal(p_estructura_red, false)
	ini_pesos_wi(red, red.entradasTamanio[1], red.entradasTamanio[2])
	ini_pesos_wo(red, red.entradasTamanio[2], red.entradasTamanio[3])
	return red
end

#	Inicializar los pesos de entrada ===================================
function ini_pesos_wi(red::RedNeuronal, filas::Int64, cols::Int64)
	for i in 1:filas
		for j in 1:cols
			red.wi[i, j] = red.funcion_calc_pesos_iniciales()
		end
	end
end

#	Inicializar los pesos de salida ====================================
function ini_pesos_wo(red::RedNeuronal, filas::Int64, cols::Int64)
	for i in 1:filas
		for j in 1:cols
			red.wo[i, j] = red.funcion_calc_pesos_iniciales()
		end
	end
end


#	Evalua la red ====================================================
function eval_red(red::RedNeuronal, inputs::Vector{Float64})
	propagar_hacia_adelante(red, inputs)
end





# 	PROPAGACIÓN: Realizar el entrenamiento de la red =================
#	==================================================================
function propagar(red::RedNeuronal, 
				  inputs::Vector{Float64},
				  outputs::Vector{Float64}
				  )

	propagar_hacia_adelante(red, inputs)
	propagar_hacia_atras(red, outputs)

end


# 	PROPAGACIÓN HACIA ADELANTE =======================================
#	==================================================================
function propagar_hacia_adelante(red::RedNeuronal, inputs::Vector{Float64})
	#Se ajusta la entra con las neuronas iniciales
	for k in 1:length(red.nodosEntrada)
		red.nodosEntrada[k] = inputs[k]
	end
	
	#Se activa la capa oculta
	for i in 1:length(red.nodosOcultos)
		sum = 0.0
		for j in 1:length(red.nodosEntrada)
			sum = sum + red.nodosEntrada[j] * red.wi[j,i]
		end
		red.nodosOcultos[i] = red.funcion_propagacion(sum)
	end
	
	#Se activa la capa de salida
	for i in 1:length(red.nodosSalida)
		sum = 0.0
		for j in 1:length(red.nodosOcultos)
			sum = sum + red.nodosOcultos[j] * red.wo[j, i]
		end
		red.nodosSalida[i] = red.funcion_propagacion(sum)
	end
end


# 	PROPAGACIÓN HACIA ATRÁS ==========================================
#	==================================================================
function propagar_hacia_atras(red::RedNeuronal, salida_esperada::Vector{Float64})
	
	#Se calculan los valores delta de la salida
	error = salida_esperada - red.nodosSalida
	deltas_salida = Array(Float64, 1, length(error))
	for i in 1:red.entradasTamanio[3]
		deltas_salida[i] = red.derivada_funcion_propagacion(red.nodosSalida[i]) * error[i]
	end
	
	#Se calculan los valores delta internas (capa oculta)
	deltas_ocultos = zeros(red.entradasTamanio[2])
	for i in 1:red.entradasTamanio[2]
		error = 0.0
		for j in 1:red.entradasTamanio[3]
			error = error + deltas_salida[j] * red.wo[i, j]
		end
		deltas_ocultos[i] = red.derivada_funcion_propagacion(red.nodosOcultos[i]) * error
	end
	
	#Se actualizan los pesos - Salida / Entrada
	for j in 1:red.entradasTamanio[2]
		for k in 1:red.entradasTamanio[3]
			cambio = deltas_salida[k] * red.nodosOcultos[j]
			red.wo[j, k] = red.wo[j, k] + red.ritmo_aprendizaje * cambio + red.momentum * red.pesos_co[j,k]
			red.pesos_co[j, k] = cambio
		end
	end
	for i in 1:red.entradasTamanio[1]
		for j in 1:red.entradasTamanio[2]
			cambio = deltas_ocultos[j] * red.nodosEntrada[i]
			red.wi[i, j] = red.wi[i, j] + red.ritmo_aprendizaje * cambio + red.momentum * red.pesos_ci[i, j]
			red.pesos_ci[i, j] = cambio
		end
	end
end


in1 = 1.0 * [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
out1 = 1.0 * [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

in2 = 1.0 * [0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
out2 = 1.0 * [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

in3 = 1.0 * [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0]
out3 = 1.0 * [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

in4 = 1.0 * [0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
out4 = 1.0 * [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

in5 = 1.0 * [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0]
out5 = 1.0 * [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

in6 = 1.0 * [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0]
out6 = 1.0 * [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

in7 = 1.0 * [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
out7 = 1.0 * [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

in8 = 1.0 * [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0]
out8 = 1.0 * [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

in9 = 1.0 * [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
out9 = 1.0 * [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

in0 = 1.0 * [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0]
out0 = 1.0 * [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]



function entrenar(output::String, oculta::Int64, max_iter::Int64)
	red = ini_red([64,oculta,10])
	for i in 1:max_iter
		propagar(red, in1, out1)
		propagar(red, in2, out2)
		propagar(red, in3, out3)
		propagar(red, in4, out4)
		propagar(red, in5, out5)
		propagar(red, in6, out6)
		propagar(red, in7, out7)
		propagar(red, in8, out8)
		propagar(red, in9, out9)
		propagar(red, in0, out0)
	end
	
	output_wi = string("out_wi", output)
	output_wo = string("out_wo", output)
	
	pesos_ci = string("out_pesos_ci", output)
	pesos_co = string("out_pesos_co", output)
	
	writedlm(output_wi, red.wi)
	writedlm(output_wo, red.wo)
	writedlm(pesos_ci, red.pesos_ci)
	writedlm(pesos_co, red.pesos_co)
	
	#return red
end


function test(red::String)
	wi = string("out_wi", red)
	wo = string("out_wo", red)
	pesos_ci = string("out_pesos_ci", red)
	pesos_co = string("out_pesos_co", red)
	net = ini_red([64,10,10])
	
	net.wi = readdlm(wi)
	net.wo = readdlm(wo)
	net.pesos_ci = readdlm(pesos_ci)
	net.pesos_co = readdlm(pesos_co)
	
	exito = 0
	
	eval_red(net, in1)
	queNumeroEs = findmax(net.nodosSalida)[2]
	if (queNumeroEs == 1) 
		exito = exito + 1 
	end
	
	eval_red(net, in2)
	queNumeroEs = findmax(net.nodosSalida)[2]
	if (queNumeroEs == 2) exito = exito + 1 end
	
	eval_red(net, in3)
	queNumeroEs = findmax(net.nodosSalida)[2]
	if (queNumeroEs == 3) exito = exito + 1 end
	
	eval_red(net, in4)
	queNumeroEs = findmax(net.nodosSalida)[2]
	if (queNumeroEs == 4) exito = exito + 1 end
	
	eval_red(net, in5)
	queNumeroEs = findmax(net.nodosSalida)[2]
	if (queNumeroEs == 5) exito = exito + 1 end
	
	eval_red(net, in6)
	queNumeroEs = findmax(net.nodosSalida)[2]
	if (queNumeroEs == 6) exito = exito + 1 end
	
	eval_red(net, in7)
	queNumeroEs = findmax(net.nodosSalida)[2]
	if (queNumeroEs == 7) exito = exito + 1 end
	
	eval_red(net, in8)
	queNumeroEs = findmax(net.nodosSalida)[2]
	if (queNumeroEs == 8) exito = exito + 1 end
	
	eval_red(net, in9)
	queNumeroEs = findmax(net.nodosSalida)[2]
	if (queNumeroEs == 9) exito = exito + 1 end
	
	eval_red(net, in0)
	queNumeroEs = findmax(net.nodosSalida)[2]
	if (queNumeroEs == 10) exito = exito + 1 end
	
	
	println(string("Casos correctos: ", exito))


end


#entrenar("hola.txt", 10, 5000)








