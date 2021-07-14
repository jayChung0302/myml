import sys, os
from datetime import datetime
import pickle

class Script:
    def __init__(self):
        self.content = ""
        self.hashtag = []
    
    def add_hashtag(self, name):
        self.hashtag.append(name)
    
    def del_hashtag(self, name):
        pass
    
class Memo:
    def __init__(self):
        self.memo = ""
        self.datelist = []
        self.name = "./auto_saved_memo"
        self.last_name = ""
        self.today = self.set_today()
    
    def set_today(self):
        return str(datetime.today()).split()[0]

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
    print()
