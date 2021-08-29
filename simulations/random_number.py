# generate a random number from [0, m) 
class RandomNumber: 
    def __init__(self, num, seed): 
        # refer to parameters in https://en.wikipedia.org/wiki/Linear_congruential_generator#Parameters_in_common_use
        self.m = num 
        self.x = seed 
        self.a = 65539
        self.c = 0
    
    def get_num(self): 
        self.x = (self.a * self.x + self.c) % self.m 
        return self.x
