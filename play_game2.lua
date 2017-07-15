--package.path = package.path .. ';/home/yanivyuval/Raphael_DQN/dqn?.lua'

require 'cudnn'
require 'alewrap'
gd = require "gd"

if not dqn then
    require "initenv"
end

--- General setup.
_opt = {}
_opt.game_path = "../roms/"
_opt.env ="breakout"
_opt.verbose = 1
_opt.actrep  = 4 
_opt.random_starts  = 1

game_env = alewrap.GameEnvironment(_opt)
game_actions = game_env:getActions()
teacher_agent = torch.load('DQN3_0_1_breakout_FULL_Y.t7')
teacher_agent = teacher_agent.agent
teacher_net = teacher_agent.network
cudnn.convert(teacher_net,cudnn)
teacher_net:cuda()

--------------------------------------------------------------------------------------------------------------------------------

--require 'initenv'
require 'randomkit'
require 'forward_nn'
data = torch.load('breakout_essai_transitions04_05.t7')
--teacher_agent = torch.load('DQN3_0_1_breakout_FULL_Y.t7')
--teacher_net = teacher_agent['model']
--cudnn.convert(teacher_net,cudnn)
--teacher_net:cuda()
s_ = data['s']
t_ = data['t']


----------------------------------------------------------------------------------------------------------------------------------

function generate_batch(batchSize,nb_actions)
    local batch_image = torch.DoubleTensor(batchSize*nb_actions,4,7056):fill(0)
    local q_actions = torch.DoubleTensor(batchSize,4,nb_actions):fill(0)
    i = 1
    while (torch.IntTensor{i}:le(batchSize)[1]== 1) do
        --print(s:type())
        terminal = false 
        k = randomkit.randint(5,999980)
        for j=1,4*nb_actions do 
           if t_[k+j-1] == 1 then
              terminal = true 
              --batch_image[i]:fill(0)
              break
           end
        end
        if not terminal then 
           for j=1,nb_actions do 
              for l = 1,4 do
           --q_actions[i] = teacher_net:forward(batch_image[i]:cuda()):double()
            batch_image[i+j-1][l] = s_[k+j-1+l]
            --print (i)
            --print (j)
            end
               end
            i=i+1
        end
    end
    q_actions = teacher_net:forward(batch_image:cuda()):double()
    q_actions = torch.reshape(q_actions,batchSize,nb_actions,4)
    local batch_image_ = torch.DoubleTensor(batchSize,4,7056):fill(0)
    for l =0,batchSize-1 do
    batch_image_[l+1] = batch_image[nb_actions*l+1]
    --print (l)
    end 
    return batch_image_,q_actions
end


function moveFrames(frames,new_frame)

  fms = torch.DoubleTensor(4,7056):fill(0)
  fms[1] = frames[2]:clone()
  fms[2] = frames[3]:clone()
  fms[3] = frames[4]:clone()
  fms[4] = new_frame:clone()
  return fms

end



function play_game_nb_actions(net_,nb_actions,steps)
--  actions_ = torch.DoubleTensor(steps+1):fill(0)
  step = 0
  cumul_reward = 0
  nb_episode = 1
  local stateDim = 7056
  local screen, reward, terminal = game_env:newGame()
 -- win = image.display({image=screen})
        local current_actions = torch.LongTensor(3,1):fill(0)
        current_action_step = torch.LongTensor(1):fill(1)
  current_frames = torch.DoubleTensor(4,7056):fill(0)
  print("Started playing...")
  while true do
          if (step>steps) then break end
        step = step+1
--print(' The step is : ')                
--print (step)
    --print (terminal)
    
        -- choose the best action
                 if  current_action_step[1] == nb_actions or terminal then
                     current_action_step:fill(1)
     		 end
                 if current_action_step[1] == 1 then
                    --print ('hello')
                    q_actions = net_:forward(current_frames:cuda()):double()
                    --print (q_actions)     
	          --   print (q_actions:type())
                    y,action_index = q_actions:max(2)     
			--print(action_index)
		--	print(y)
	--q_actions = torch.reshape(q_actions,batchSize,nb_actions,4)
                   -- action_index =  torch.randperm(nb_actions)-- agent:perceive(reward, screen, terminal, true, 0)
                 end 
       if not terminal then
                        --print ('current_action_step[1]')
                        --print (current_action_step[1])

                        --print ('action index')

                        --print (action_index)
                        --print ('action_index[current_action_step[1]]')
                        --print (action_index[current_action_step[1]])
                        --print ('game_actions[action_index[current_action_step[1]]:squeeze()]')
                        --print (game_actions[action_index[current_action_step[1]]:squeeze()])
                        --actions_[step] = action_index[current_action_step[1]]
			
                          screen, reward, terminal = game_env:step(game_actions[action_index[current_action_step[1]]:squeeze()], false)
                        --screen, reward, terminal = game_env:step(game_actions[action_index[current_action_step[1]]:squeeze()], false)
--                          print(screen)
                          current_frames = moveFrames(current_frames,screen:clone():resize(1,7056))
                          cumul_reward = cumul_reward + reward
                          current_action_step:add(1)
       else
                nb_episode  = nb_episode + 1
                current_frames = torch.DoubleTensor(4,7056):fill(0)
                if _opt.random_starts > 0 then
                      screen, reward, terminal = game_env:nextRandomGame()
                      current_frames = moveFrames(current_frames,screen:clone():resize(1,7056))
                      cumul_reward  = cumul_reward + reward
                else
                    screen, reward, terminal = game_env:newGame()
                    current_frames = moveFrames(current_frames,screen:clone():resize(1,7056))
                    cumul_reward  = cumul_reward + reward
                end
       end
  
          --image.display({image=screen, win=win})
          --return  actions_
	end

print ("cumul reward is :")
print (cumul_reward)
print ("nb episode is :")
print(nb_episode)
return cumul_reward/nb_episode
end



-------------------------------------------------------------------------------------------------------------------------------
nb_actions = 3
student_net = create_network(args,nb_actions)
student_net:add(nn.Reshape(nb_actions,4))


student_net = teacher_agent.network:clone()
student_net:remove()
student_net:add(nn.Linear(512,4*nb_actions))
student_net:add(nn.Reshape(nb_actions,4))


kullback =  false 


if kullback then
	student_net:add( nn.LogSoftMax())
end
cudnn.convert(student_net,cudnn)
student_net:cuda()
criterion_MSE = nn.MSECriterion():cuda()
criterion_KB  = nn.DistKLDivCriterion():cuda()
parallel_criterion = nn.ParallelCriterion(false):cuda()
for i=1,nb_actions do
	parallel_criterion:add(nn.MSECriterion():cuda())
end

w,dE_dw = student_net:getParameters()	
require 'optim'
batchSize = 128

optimState = {
    --learningRate = 0.1
}



function forwardNet(kullback,train)

    local lossAcc = 0
    local numBatches = 1000 --100 --2500 100
    if train then
        --set network into training mode
       student_net:training()
    end
    for i = 1, numBatches do
	--print(i)
       	x,yt = generate_batch(batchSize,nb_actions)
        yt:cuda()
        x:cuda()
        local y = student_net:forward(x:cuda())
        y:cuda()
        --print (y)
        --print (yt)

        --err = criterion_MSE:forward(y:float():cuda(), yt:float():cuda())
        err = parallel_criterion:forward(y:float():cuda(), yt:float():cuda())
        lossAcc = lossAcc + err
        
        if train then
            function feval()
                student_net:zeroGradParameters() --zero grads

             --local dE_dy = criterion_MSE:backward(y:float():cuda(), yt:float():cuda())
		local dE_dy = parallel_criterion:backward(y:float():cuda(), yt:float():cuda())
                student_net:backward(x, dE_dy) -- backpropagation
            
                return err, dE_dw
            end
            --w_ = w:clone()
            optim.rmsprop(feval, w, optimState)
	    --print ("difference in weights :")
           -- print(w-w_)
        end
    end

    local avgLoss = lossAcc / numBatches
    
    return avgLoss
end



epochs = 1000
--trainLoss = torch.Tensor(epochs)
--testLoss = torch.Tensor(epochs)
--trainError = torch.Tensor(epochs)
mean_reward = torch.Tensor(epochs):fill(0)
trainLoss = torch.Tensor(epochs):fill(0)

--x,y = generate_batch(5,2)
--print (x:size())
--print (y:size())

--reset net weights
--student_net:apply(function(l) l:reset() end)
--wrap_agent = torch.load('DQN3_0_1_breakout_FULL_Y_FOR_ER.t7')
--wrap_agent = torch.load('DQN3_0_1_breakout_FULL_Y.t7')
--wrap_agent = wrap_agent.agent
--wrap_agent.nb_actions = nb_actions



filename ="breakout_distillation_results_1000_by_epochs.t7"


for e = 1, epochs do   
    print('Epoch ' .. e .. ':')
    trainLoss[e]  = forwardNet(false,true)
    print ("train loss is:  " .. trainLoss[e]) 
    if e%1== 0 then
    --trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
   -- trainLoss[e] ,trainError[e] = forwardNet(s, a, true)
    --trainLoss[e]  = forwardNet(s, a, true)
    
    --student_net =  student_agent.network 

--- lignes suivantes A decommenter
   --  wrap_agent.network = student_net:clone()
     mean_reward[e] = play_game_nb_actions(student_net,nb_actions,100)
     
    --print('Mean reward: ' .. mean_reward[e])
    end
        if e%5== 0 then
    torch.save(filename,{agent=student_net,trainLoss=trainLoss,mean_reward=mean_reward})
end
end
