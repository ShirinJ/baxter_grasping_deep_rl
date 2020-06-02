local nn = require 'nn'
local torches = require 'torch'
require 'dpnn'
require 'classic.torch' -- Enables serialisation

local Body = classic.class('Body')

-- Architecture based on "Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning and Large-Scale Data Collection"

local histLen = 4
local numFilters = 32
local motorInputs = 4
local batchSize = 32
local size = 60

local data = torch.FloatTensor(batchSize, histLen, 7, size, size):uniform() -- Minibatch
data[{{}, {}, {4}, {}, {}}]:zero() -- Zero motor inputs
data[{{}, {}, {4}, {1}, {1, motorInputs}}] = 2 -- 3 motor inputs

function Body:_init(opts)
  opts = opts or {}
end

function Body:createBody()
	local net = nn.Sequential()
	net:add(nn.View(-1, histLen, 7, size, size))
	
	local imageNet = nn.Sequential()
	imageNet:add(nn.Narrow(3, 1, 3)) -- Extract 1st 3 (RGB) channels
	imageNet:add(nn.View(histLen * 3, size, size):setNumInputDims(4)) -- Concatenate in time
	imageNet:add(nn.SpatialConvolution(histLen * 3, numFilters, 7, 7, 3, 3))
	imageNet:add(nn.ReLU(true))
	imageNet:add(nn.SpatialConvolution(numFilters, numFilters, 5, 5, 2, 2))
	imageNet:add(nn.ReLU(true))
	    
	local depthNet = nn.Sequential()
	depthNet:add(nn.Narrow(3, 5, 3)) -- Extract 5th - 7th channels
	depthNet:add(nn.View(histLen * 3, size, size):setNumInputDims(4))
	depthNet:add(nn.SpatialConvolution(histLen * 3, numFilters, 7, 7, 3, 3))
	depthNet:add(nn.ReLU(true))
	depthNet:add(nn.SpatialConvolution(numFilters, numFilters, 5, 5, 2, 2))
	depthNet:add(nn.ReLU(true))
	depthNet:add(nn.SpatialDropout(0))
	
	local branches = nn.ConcatTable()
	branches:add(imageNet)
	branches:add(depthNet)
	
	local RGBDnet = nn.Sequential()
	RGBDnet:add(branches)
	RGBDnet:add(nn.JoinTable(2, 4))
	RGBDnet:add(nn.SpatialConvolution(numFilters * 2, numFilters, 3, 3)) --1st param *2 for RGBRGB
	RGBDnet:add(nn.ReLU(true))
	RGBDnet:add(nn.View(1):setNumInputDims(3)) --unroll
	
	local motorNet = nn.Sequential()
	motorNet:add(nn.Narrow(3, 4, 1)) -- Extract 4th channel
	motorNet:add(nn.View(histLen, size, size):setNumInputDims(4)) -- Concatenate in time
	motorNet:add(nn.Narrow(3, 1, 1)) -- Extract motor inputs row
	motorNet:add(nn.Narrow(4, 1, motorInputs)) -- Extract motor inputs
	motorNet:add(nn.View(histLen * motorInputs):setNumInputDims(3)) -- Unroll
	motorNet:add(nn.Linear(histLen * motorInputs, numFilters)) 
	motorNet:add(nn.View(1):setNumInputDims(1))
	
	
	local merge = nn.ConcatTable()
	merge:add(RGBDnet)
	merge:add(motorNet)
		
	
	net:add(merge)
	net:add(nn.JoinTable(1,2))
	--[[
	--print network architecture 
	for i,mod in ipairs(net:listModules()) do
		print(mod)
	end--]]
	return net
end

return Body
