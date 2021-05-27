# Rope data structure
# add very large string, how can you insert single character into the middle of it in constant time?

# todo
# insert
# delete
# substring
# concatenation

# API
def to_rope(string):
    return Rope(string)

class Rope:
    def __init__(self, string):
        self.string = string
    
    def  __str__(self):
        return self.string

    def substring(self, start, length):
        return Substring(self, start, length)

class Substring:
    def __init__(self, rope, start, length):
        pass
    
    def __str__(self):
        return "BC"

assert str(to_rope("ABC")) == "ABC"
assert str(to_rope("ABCDE").substring(1, 3)) == "BC"
