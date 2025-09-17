class Rewards():
    def alive(self, env)->tuple[float, bool, dict]: 
        return 0.0, False, False, {"steps_since_food": env.steps_since_food}
    
    def foodReward(self, env)->tuple[float, bool, dict]:
        return 2.0, False, False, {"steps_since_food": env.steps_since_food}
    
    def winReward(self, env)->tuple[float, bool, dict]:
        return 5.0, True, False, {"steps_since_food": env.steps_since_food}

    #TODO    
    # def distanceReward(self, env)->tuple[float, bool, dict]:
    #     return 2.0, False, {}


    def deathPenalty(self, env)->tuple[float, bool, dict]:
        return -1.0, True, False, {"steps_since_food": env.steps_since_food}
    
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