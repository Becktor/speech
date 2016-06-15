import sys
class progressBar:
    def __init__(self, max):
        self.max = abs(max/100)
        self.tenth = abs(max/10)
        self.ones = 1
        self.tens = 1

    def p100(self,cntr):
        if(self.max * self.ones < cntr and self.ones < 101):
            sys.stdout.write("\r{0}>".format(str(self.ones) + "% " + "=" * self.tens))
            sys.stdout.flush()
            self.ones += 1
            if(self.tenth * self.tens < cntr):
                self.tens+=1
