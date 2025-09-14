class Rewards():
    def alive(env)->tuple[float, bool, dict]: 
        return 0.0, False, {}
    
    def foodReward(env)->tuple[float, bool, dict]:
        return 2.0, False, {}
    
    def winReward(env)->tuple[float, bool, dict]:
        return 5.0, False, {}

    #TODO    
    # def distanceReward(env)->tuple[float, bool, dict]:
    #     return 2.0, False, {}


    def deathPenalty()->tuple[float, bool, dict]:
        return -1.0, False, {}
    
    #TODO
    # def distancePenalty(env)->tuple[float, bool, dict]:
    #     return 2.0, False, {} 
    #TODO
    # def too_many_steps(env)->tuple[float, bool, dict]:
    #     return 2.0, False, {}
    