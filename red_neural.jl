

type RedNeuronal

	dimensiones::Array{Int64, 1}
	usa_bias::Bool
	ritmo_aprendizaje::Float64
	momentum::Float64
	
	nodosEntrada::Array{Float64, 1}
	nodosOcultos::Array{Float64, 1}
	nodosSalida::Array{Float64, 1}
	
	wi::Array{Array{Float64,1},1}					# wi - Pesos de entrada
	wo::Array{Array{Float64,1},1}					# wo - Pesos de salida
	
	pesos::Array{Array{Float64,1},1}				# ci
	deltas::Array{Array{Float64,1},1}				# co
	
	funcion_calc_pesos_iniciales::Function
	funcion_propagacion::Function
	derivada_funcion_propagacion::Function

end

#	Función para crear una matriz inicializada
function crearMatriz(cant_filas, cant_columnas)

end

#	Función constructura para la red neuronal
function RedNeuronal(p_estructura::Array{Int64, 1}, p_usa_bias::Bool)
	
	dim_red = p_estructura
	dim_pesos = length(p_estructura) - 1
	
	
	RedNeuronal(p_estructura,						#	Tamaño de la red neuronal
				p_usa_bias,							#	Indica si usa bias
				0.25,								#	Ritmo de aprendizaje
				0.1,								#	Momentum
				rand(dim_red[1]),					#	Nodos de entrada
				rand(dim_red[2]),					#	Nodos de la capa oculta
				rand(dim_red[3]),					#	Nodos de la capa de salida
				[],									#	Pesos de entrada
				[],									#	Pesos de salida
				[],									#	Pesos desde la capa de entrada hasta ocultos
				[],									#	Pesos deltas
				() -> rand(-2.0:2.0),
				(x::Float64) -> 1/(1+exp(-1*(x))),
				(y::Float64) -> y*(1-y)
				)			
end


function ini_red(p_estructura_red::Array{Int64,1})
	red = RedNeuronal(p_estructura_red, false)
	return red
end