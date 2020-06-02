ros = require 'ros'
ros.init('DQN_comms')
local classic = require 'classic'
require 'image'

local bit = require("bit")

torch.setdefaulttensortype('torch.FloatTensor')

local BaxterEnv, super = classic.class('BaxterEnv', Env)
-- local BaxteEnv = {};

local img_resp_ready = false
local dep_resp_ready = false
local raw_msg = torch.FloatTensor(14400)
local dep_raw_msg = torch.FloatTensor(14400)
local task = 0
local step = 1

function connect_cb(name, topic)
  print("subscriber connected: " .. name .. " (topic: '" .. topic .. "')")
end


function disconnect_cb(name, topic)
  print("subscriber diconnected: " .. name .. " (topic: '" .. topic .. "')")
end

function sleep(n)
  os.execute("sleep " .. tonumber(n))
end
-- Constructor
function BaxterEnv:_init(opts)
	opts = opts or {}

	--setup state variables
	self.img_size = 60
	self.screen = torch.FloatTensor(7,self.img_size,self.img_size):zero()

	--setup ros node and spinner (processes queued send and receive topics)
	self.spinner = ros.AsyncSpinner()
	self.spinner:start()
	self.nodehandle = ros.NodeHandle()
	
	--Message Formats
	self.string_spec = ros.MsgSpec('std_msgs/String')
	self.image_spec = ros.MsgSpec('sensor_msgs/Image')

	-- Create publisher
	self.publisher = self.nodehandle:advertise("chatter",
	self.string_spec, 100, false, connect_cb, disconnect_cb)
	ros.spinOnce()

	--create subscriber
	self.timeout = 20.0
	
	self.img_subscriber = self.nodehandle:subscribe("baxter_view", self.image_spec, 		100, 		{ 'udp', 'tcp' }, { tcp_nodelay = true })
	
	self.dep_subscriber = self.nodehandle:subscribe("DepthMap", self.image_spec, 		100, 		{ 'udp', 'tcp' }, { tcp_nodelay = true })

	--Received message callback
	self.img_subscriber:registerCallback(function(msg, header)
		local frame_id = msg.header.frame_id
		-- get raw message data & task status (0 for incomplete 1 for complete)
		raw_msg = msg.data
		task = tonumber(frame_id)
		img_resp_ready = true
	end)
	
	self.dep_subscriber:registerCallback(function(msg)
		-- get raw message data
		dep_raw_msg = msg.data
		dep_resp_ready = true
	end)
	
	--spin must be called continually to trigger callbacks for any received messages
	ros.spinOnce()
	img_resp_ready = false
	dep_resp_ready = false
end


function BaxterEnv:sendMessage(message)
	--Wait for a subscriber
	local t = os.clock()
	while self.publisher:getNumSubscribers() == 0 do
		if (os.clock() - t) < self.timeout then
			io.write(string.format("waiting for subscriber: %.2f \r",(os.clock() -t)))
		else
			io.write("Wait for subscriber timed out \n")
			return false
		end
	end

	-- Send Message
	img_resp_ready=false
	dep_resp_ready = false
	m = ros.Message(self.string_spec)
	m.data = message
	self.publisher:publish(m)
	return true
end

-- Wait for response message to be recieved - sending message is blocking function
function BaxterEnv:waitForResponse(message)
	local t = os.clock()
	local tries = 1
	while not (dep_resp_ready and img_resp_ready) do 
		if (os.clock() - t) < self.timeout then
			sys.sleep(0.1)
			ros.spinOnce()
		else
			if tries > 3 then
				print("Exceeded number of attempts \n")
				self:_close()
				return false
			else
				print("Robot response timed out. Resending " .. message .. "\n")
				self.sendMessage(message)
				ros.spinOnce()
				t = os.clock()
				tries = tries + 1
			end
		end
	end
	return true
end

function BaxterEnv:msgToImg()
	-- Sort message data - pixel values come through in order r[1], g[1], b[1], a[1], r[2], b[2], g[2], .. etc with the alpha channel representing motor angle information
	-- Each depth pixel is 2 byte unsigned short format in last two channels (small endian)
	local reordered_Data = torch.FloatTensor(14400)
	local reordered_dep = torch.FloatTensor(10800)
	
	for i = 1, 14400 do
		if i%4==1 then
			reordered_Data[(i+3)/4] = raw_msg[i]/255
			reordered_dep[(i+3)/4] = dep_raw_msg[i]/255
		elseif i%4==2 then
			reordered_Data[3600 + (i+2)/4] = raw_msg[i]/255
			reordered_dep[3600 + (i+2)/4] = dep_raw_msg[i]/255
		elseif i%4 == 3 then
			reordered_Data[7200 + (i+1)/4] = raw_msg[i]/255
			reordered_dep[7200 + (i+1)/4] = dep_raw_msg[i]/255
		else
			reordered_Data[10800 + i/4] = raw_msg[i]/255
		end
	end

	self.screen[{ {1,4},{},{} }] = torch.reshape(reordered_Data, 4, self.img_size,self.img_size)
	self.screen[{ {5,7},{},{} }] = torch.reshape(reordered_dep, 3, self.img_size,self.img_size)
end

-- 1 state returned, of type 'int', of dimensionality 6 x self.img_size x self.img_size, between 0 and 1
function BaxterEnv:getStateSpec()
	return {'real', {7, self.img_size, self.img_size}, {0, 1}}
end

-- 1 action required, of type 'int', of dimensionality 1, between 0 and 6
function BaxterEnv:getActionSpec()
	return {'int', 1, {0, 4}}
end

-- RGB screen of size self.img_size x self.img_size
function BaxterEnv:getDisplaySpec()
	return {'real', {3, self.img_size, self.img_size}, {0, 1}}
end

-- Min and max reward
function BaxterEnv:getRewardSpec()
	return 0, 0.01, -0.015, 0.1, -0.1, 1, -1
	-- erroreous, correct dir, wrong dir, contact obj, failed pick up, success, timeout
end


-- Starts new game
function BaxterEnv:start()
	self:sendMessage("reset")
	self:waitForResponse("reset")
	sleep(0.1)
	ros.spinOnce()
	self:msgToImg()
-- Return observation
	return self.screen
end

-- Steps in a game
function BaxterEnv:step(action)
	-- Reward is 0 by default
	local reward = 0
	local terminal = false
	-- Move player - 0 corresponds to picking up object
	-- Task ends once an attempt to pick up the object is made
	-- This is because unsuccesful attempts often knock the block away
	-- making it impossible to pickup again
	-- 1-6 control +ve or -ve for 3DOF - shoulder, wrist and elbow
	
	if action == 0 then
		terminal = true
		step = 1
	end
	-- Send action to gazebo
	self:sendMessage(tostring(action))
	self:waitForResponse(tostring(action))
	-- get next message
	self:msgToImg()
	-- Check task condition
	if step == 30 then
		terminal = true
		step = 1
		reward = -1
	else
		reward = task
	end
	
	step = step + 1
	self.signal = reward --testing use
	return reward, self.screen, terminal
end

-- Returns (RGB) display of screen
function BaxterEnv:getDisplay()
	return torch.repeatTensor(self.screen, 3, 1, 1)
end

function BaxterEnv:_close()
	self.subscriber:shutdown()
	ros.shutdown()
end

return BaxterEnv

