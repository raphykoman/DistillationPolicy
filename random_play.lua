require 'torch'
require 'alewrap'
require 'image'
require 'gnuplot'





-----------------------Functions----------------------------

function moveFrames(frames,new_frame)

  fms = torch.DoubleTensor(4,7056):fill(0)
  fms[1] = frames[2]:clone()
  fms[2] = frames[3]:clone()
  fms[3] = frames[4]:clone()
  fms[4] = new_frame:clone()
  return fms

end


-----------------------------------------------------------

_opt = {}
_opt.game_path = "../roms/"
_opt.env ="breakout"
_opt.verbose = 1
_opt.actrep  = 4 
_opt.random_starts  = 1

game_env = alewrap.GameEnvironment(_opt)
game_actions = game_env:getActions()








local screen, reward, terminal = game_env:newGame()

local previm = im
local win = image.display({image=screen})

print("Started playing...")

-- play one episode (game)
step = 0
while step < 10 do
    -- choose the best action
    local action_index = torch.random(1,table.getn(game_actions)) -- agent:perceive(reward, screen, terminal, true, 0.05)

    -- play game in test mode (episodes don't end when losing a life)
    screen, reward, terminal = game_env:step(game_actions[action_index], false)
    if terminal then 
    	game_env:newGame()
    end

    -- display screen
    image.display({image=screen, win=win})

	step = step +1
end





function play_game_nb_actions(nb_actions,steps)
	actions_ = torch.DoubleTensor(steps+1):fill(0)
	step = 0
	cumul_reward = 0
	nb_episode = 1
	local stateDim = 7056
	local screen, reward, terminal = game_env:newGame()
	local win = image.display({image=screen})
        local current_actions = torch.LongTensor(3,1):fill(0)
        current_action_step = torch.LongTensor(1):fill(1)
	print("Started playing...")
	while true do
        	if (step>steps) then break end
    		step = step+1
                --print (step)
		--print (terminal)
    
    		-- choose the best action
                 if  current_action_step[1] == nb_actions or terminal then
                    current_action_step:fill(1)
		 end
                 if current_action_step[1] == 1 then
                    --print ('hello')
   		    action_index =  torch.randperm(nb_actions)-- agent:perceive(reward, screen, terminal, true, 0)
                 end 
   		 if not terminal then
   		 			--print ('current_action_step')
                    --print (current_action_step)
                    --print ('action index')
                    --print (action_index)
                    --print ('action_index[current_action_step[1]]')
                    --print (action_index[current_action_step[1]])
                    --print ('game_actions[3]')
		     		--print (game_actions[3])
		     		--print ('game_actions[action_index[current_action_step[1]]:squeeze()]')
                    --print (game_actions[action_index[current_action_step[1]]])
                    actions_[step] = action_index[current_action_step[1]]
  		      		screen, reward, terminal = game_env:step(game_actions[action_index[current_action_step[1]]], false)
  		        	--screen, reward, terminal = game_env:step(game_actions[action_index[current_action_step[1]]:squeeze()], false)

      		      cumul_reward = cumul_reward + reward
                      current_action_step:add(1)
   		 else
     		      nb_episode  = nb_episode + 1
        	      if _opt.random_starts > 0 then
                      screen, reward, terminal = game_env:nextRandomGame()
                      cumul_reward  = cumul_reward + reward
        	      else
            	      screen, reward, terminal = game_env:newGame()
            	      cumul_reward  = cumul_reward + reward
                      end
                  end
	
          image.display({image=screen, win=win})
          --return  actions_
end

print ("cumul reward is :")
print (cumul_reward)
print ("nb episode is :")
print(nb_episode)
return cumul_reward/nb_episode
end


play_game_nb_actions(3,1000)

--gnuplot.hist(actions_)









