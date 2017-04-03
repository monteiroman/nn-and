require 'torch'
require 'nn'

mlp = nn.Sequential()  -- make a multi-layer perceptron
inputs = 2; outputs = 1; HUs = 10 -- parameters
mlp:add(nn.Linear(inputs, HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs, outputs))
--mlp:add(nn.Linear(inputs, outputs))

criterion = nn.MSECriterion()

function train(net, criterion, trainer, iter)
	local input, output = trainer(iter)

	-- feed it to the neural network and the criterion
	result = net:forward(input)
	criterion:forward(result, output)

	-- train over this example in 3 steps
	-- (1) zero the accumulation of the gradients
	net:zeroGradParameters()
	-- (2) accumulate gradients
	d_df = criterion:backward(net.output, output)
	net:backward(input, d_df)
	-- (3) update parameters with a 0.01 learning rate
	net:updateParameters(0.01)

	diffTensor = output - result
	return math.abs(diffTensor[1])
end

trainingEpochs = 2*1e4
for i = 1, trainingEpochs do
	io.write('\r' .. i .. '/' .. trainingEpochs)

	train(mlp, criterion, function (iter)
		-- random sample
		local input = torch.randn(2)     -- normally distributed example in 2d
		if input[1]<0.5 then input[1]=0 else input[1]=1 end
		if input[2]<0.5 then input[2]=0 else input[2]=1 end
		local output = torch.Tensor(1)

		--AND VALUES
		if input[1]==0 and input[2]==0 then output[1]=0 end
		if input[1]==0 and input[2]==1 then output[1]=0 end
		if input[1]==1 and input[2]==0 then output[1]=0 end
		if input[1]==1 and input[2]==1 then output[1]=1 end


		return input, output
	end, i)
end
print(' - Done!')

params = mlp:getParameters()
print(params)

function test(a, b)
	x = torch.Tensor(2)
	x[1] = a
	x[2] = b
	c = mlp:forward(x)
	print(a .. ' and ' .. b .. ' = ' .. c[1])
end

test(0, 0)
test(0, 1)
test(1, 0)
test(1, 1)
test(0, 0)
