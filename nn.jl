type RedNeuronal
    structure::Array{Int64, 1}
    disable_bias::Bool
    ritmo_aprendizaje::Float64
    momentum::Float64
    initial_weight_function::Function
    propagation_function::Function
    derivative_propagation_function::Function
    activation_nodes::Array{Array{Float64}, 1}
    weights::Array{Array{Float64}, 1}
    last_changes::Array{Array{Float64}, 1}
    deltas_salida::Array{Array{Float64}, 1}
end

function RedNeuronal(structure::Array{Int64, 1}, disable_bias::Bool)
    len_struct  = length(structure)
    len_weights = length(structure) - 1

    RedNeuronal(structure,
                  disable_bias,
                  0.25,
                  0.1,
                  () -> rand(0:2000)/1000.0 - 1,
                  (x::Float64) -> 1/(1+exp(-1*(x))),
                  (y::Float64) -> y*(1-y),
                  Array(Array{Float64}, len_struct),
                  Array(Array{Float64}, len_weights),
                  Array(Array{Float64}, len_weights),
                  Array(Array{Float64}, 1)
                  )
end

function init_network(structure::Array{Int64,1})
    red = RedNeuronal(structure, false)
    init_activation_nodes(red)
    init_weights(red)
    init_last_changes(red)
    return red
end

function init_activation_nodes(red::RedNeuronal)
    len = length(red.activation_nodes)
    # for each layer in red, build 1.0 matrices
    for i in 1:len
        if !red.disable_bias && i < len
            red.activation_nodes[i] = ones(red.structure[i] + 1)
        else
            red.activation_nodes[i] = ones(red.structure[i])
        end
    end
end

function init_weights(red::RedNeuronal)
    for i in 1:length(red.weights)
        arr = Array(Float64, length(red.activation_nodes[i]), red.structure[i+1])

        for j=1:length(arr)
            arr[j] = red.initial_weight_function()
        end

        red.weights[i] = arr
    end
end

function init_last_changes(red::RedNeuronal)
    for i in 1:length(red.last_changes)
        red.last_changes[i] = zeros(size(red.weights[i]))
    end
end

function train(red::RedNeuronal, inputs::Vector{Float64}, outputs::Vector{Float64})

	for i in 1:10000
		net_eval(red, inputs)
		backpropagate(red, outputs)
		calculate_error(red, outputs)
	end
    
end

function net_eval(red::RedNeuronal, inputs::Vector{Float64})
    check_input_dimension(red, inputs)
    if length(red.weights) == 0
        init_network(red)
    end
    feedforward(red, inputs)
    return red.activation_nodes[end]
end

function feedforward(red::RedNeuronal, inputs::Vector{Float64})
    for i in 1:length(inputs)
        red.activation_nodes[1][i] = inputs[i]
    end

    for n in 1:length(red.weights)
        for j in 1:red.structure[n+1]
            s = dot(red.activation_nodes[n], red.weights[n][:, j])
            red.activation_nodes[n+1][j] = red.propagation_function(s)
        end
    end
end

function backpropagate(red::RedNeuronal, expected_values::Vector{Float64})
    check_output_dimension(red, expected_values)
    calculate_output_deltas(red, expected_values)
    calculate_internal_deltas(red)
    update_weights(red)
end

function calculate_output_deltas(red::RedNeuronal, expected_values::Vector{Float64})
    output_values = red.activation_nodes[end]
    err = expected_values - output_values
    output_deltas = Array(Float64, 1, length(err))
    for i=1:length(err)
        output_deltas[i] = red.derivative_propagation_function(output_values[i]) * err[i]
    end
    red.deltas_salida = Array{Float64}[output_deltas]
end

function calculate_internal_deltas(red::RedNeuronal)
    prev_deltas = red.deltas_salida[end]
    for layer_index=2:length(red.activation_nodes)-1
        layer_deltas = Array(Float64,1,length(red.activation_nodes[layer_index]))
        for j=1:length(red.activation_nodes[layer_index])
            err = 0.0
            for k=1:red.structure[layer_index+1]
                err += prev_deltas[k] * red.weights[layer_index][j,k]
            end
            layer_deltas[j] = red.derivative_propagation_function(red.activation_nodes[layer_index][j]) * err
        end
        unshift!(red.deltas_salida, layer_deltas)
    end
end

function update_weights(red::RedNeuronal)
    for n=1:length(red.weights)
        for i=1:size(red.weights[n],1)
            for j=:1:size(red.weights[n],2)
                change = red.deltas_salida[n][j] * red.activation_nodes[n][i]
                red.weights[n][i,j] += (red.ritmo_aprendizaje * change + red.momentum * red.last_changes[n][i,j])
                red.last_changes[n][i,j] = change
            end
        end
    end
end

function calculate_error(red::RedNeuronal, expected_output::Vector{Float64})
    output_values = red.activation_nodes[end]
    err = 0.0
    diff = output_values - expected_output
    for output_index=1:length(diff)
        err +=
        0.5 * diff[output_index]^2
    end
    return err
end

# TODO: throw exception here..
function check_input_dimension(red::RedNeuronal, inputs::Vector{Float64})
    if length(inputs) != red.structure[1]
        error("Wrong number of inputs.\n",
        string("Expected: ", red.structure[1], "\n"),
        string("Received: ", length(inputs)))
    end
end

function check_output_dimension(red::RedNeuronal, outputs::Vector{Float64})
    if length(outputs) != red.structure[end]
        error("Wrong number of outputs.\n",
        string("Expected: ", red.structure[end], "\n"),
        string("Received: ", length(outputs)))
    end
end



in1 = 1.0 * [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0
]
out1 = 1.0 * [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

in9 = 1.0 * [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0
]
out9 = 1.0 * [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

red = init_network([64,10,10])
train(red, in9, out9)
train(red, in1, out1)


