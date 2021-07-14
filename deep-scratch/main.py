from memo import Script, Memo
import sys

def process():
    memo_name = input("file name?")
    scpt = Script(memo_name)
    print('memo whatever you want,')
    x = sys.stdin.readline().rstrip()
    hash_name = input("hash tag?").split()
    scpt.add_hashtag(hash_name)
    return scpt

if __name__ == '__main__':
    memo = Memo()
    script = process()
    
    memo.put_script(script)
    memo.total_save()
    memo.show_memo()
    memo.load_memo('./auto_saved_memo.pkl')
