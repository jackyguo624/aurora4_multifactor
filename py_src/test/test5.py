
class AIter(object):
    def __init__(self, n):
        self.n = n
    def __iter__(self):
        return self
    def __next__(self):
        if self.n < 10:
            self.n += 1
            return self.n
        else:
            raise StopIteration


aiter = AIter(0)

for _ in range(100):
    print(next(aiter))
