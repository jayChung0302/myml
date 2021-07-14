import sys, os
import datetime
import pickle

class Memo:
    def __init__(self):
        self.memo = ""
        self.datelist = []
        self.name = "./auto_saved_memo.txt"
        self.last_name = ""
        self.today = self.today()
    
    def today(self):
        pass

    def save(self, save_dir="./"):
        with open(self.name + '.pkl', 'wb') as file:
            pickle.dump(self.memo, file)

    def load(self, load_dir="./"):
        pass

    def see(self, ):
        pass

if __name__ == '__main__':
    memo = Memo()
    memo.save()
