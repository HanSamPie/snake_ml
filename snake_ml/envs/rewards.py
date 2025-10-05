# score distribution -> track score at death
#   average score

# death cause -> return string value
# death location (heatmap) -> return cords

# path fitness -> measure of how close the path was to the theoretical shortest path

# Time to score n -> steps since last fruit
# shows the increase in steps needed to reach fruit n compared to n+1

# inputs to score n -> track inputs since last apple
# can provide insights into the number of evasive maneuvers taken 
# (though a stair movement is just as effective in reducing distance as a simple 90° and are therefore not as 
# conclusive to performance and depend more on the reward function and training method)

class Rewards():
    def alive(self, env, new_head)->tuple[float, bool, dict]: 
        return 0.0, False, False, {
            "steps_since_food": env.steps_since_food,
            "score": len(env.snake),
            "death": {
                "cause": "deprecated",
                "position": new_head
            }
        }
    
    def foodReward(self, env)->tuple[float, bool, dict]:
        x_old, y_old = env.head_new_food
        x_food, y_food = env.snake[0]
        
        opt_path = abs(x_old - x_food) + abs(y_old - y_food)

        return 2.0, False, False, {
            "steps_since_food": env.steps_since_food,
            "path": {
                "optimal_path": opt_path,
                "actual_path": env.steps_since_food,
                "fruit_num": len(env.snake) - 1
            },
            "score": len(env.snake) - 1,
            "num_inputs": env.inputs_since_food
        }
    
    def winReward(self, env)->tuple[float, bool, dict]:
        x_old, y_old = env.head_new_food
        x_food, y_food = env.snake[0]
        
        opt_path = abs(x_old - x_food) + abs(y_old - y_food)
        
        return 5.0, True, False, {
            "steps_since_food": env.steps_since_food,
            "path": {
                "optimal_path": opt_path,
                "actual_path": env.steps_since_food,
                "fruit_num": len(env.snake) - 1
            },
            "score": len(env.snake) - 1,
            "num_inputs": env.inputs_since_food
        }

    #TODO    
    # def distanceReward(self, env)->tuple[float, bool, dict]:
    #     return 2.0, False, {}


    def deathPenalty(self, env, new_head: tuple)->tuple[float, bool, dict]:
        cause = ""
        if (not (0 <= new_head[0] < env.board_size and 0 <= new_head[1] < env.board_size)):
            cause = "border"
        elif new_head in env.snake:
            cause = "snake"
            
        return -1.0, True, False, {
            "steps_since_food": env.steps_since_food,
            "score": len(env.snake),
            "death": {
                "cause": cause,
                "position": new_head
            }
        }
    
    #TODO
    # def distancePenalty(self, env)->tuple[float, bool, dict]:
    #     return 2.0, False, {} 
    #TODO
    # def too_many_steps(self, env)->tuple[float, bool, dict]:
    #     return 2.0, False, {}
        # pos_new, pos_old = self.snake[:2]

        # if math.dist(pos_new, self.food) < math.dist(pos_old, self.food):
        #     reward += 0.1 #* (14-math.dist(pos_new, self.food))/14  # reward for moving closer
        # else:
        #     reward -= max_steps/(max_steps - self.steps + 1) - 1 # small penalty otherwise  