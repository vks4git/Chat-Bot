import numpy


class Loader:
    def __init__(self, vec_size):
        f = open("All-seasons.csv", "r")
        arr_fix = []
        arr_var = []
        self._vec_size = vec_size
        self._ser = {}
        self._deser = {}
        ind = 0
        for s in f.read().split("\""):
            ind += 1
            if ind % 2 == 1:
                continue
            var, fix = self.toint(s)
            arr_fix.append(fix)
        f.close()
        self._fixed_len = numpy.array(arr_fix)

    def toint(self, s):
        ans_fixed = numpy.zeros(self._vec_size, dtype="float32")
        ans_var = []
        ind = 0
        for c in s:
            if c not in self._ser.keys():
                self._ser[c] = len(self._ser) + 1
                self._deser[len(self._deser) + 1] = c
            ans_fixed[ind] = self._ser[c]
            ans_var.append(self._ser[c])
            ind += 1
        fix = numpy.array(ans_var)
        return fix, ans_fixed

    def get(self):
        return numpy.multiply(self._fixed_len, 1 / len(self._ser))

    def tostring(self, arr):
        arr = numpy.multiply(numpy.reshape(arr, self._vec_size), len(self._ser)).astype("int32")
        s = ""
        for i in arr:
            c = int(min(i, 1000))
            if c == 0:
                break
            if c not in self._deser.keys():
                s += "?"
            else:
                s += self._deser[c]
        return s
