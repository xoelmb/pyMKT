import os, signal
import numpy as np


class Memorizer:

    """
    A class for recording ram usage.
    Requires the memorizer.sh script.
    Two temporary files are used: memory.log, state.log
    Create a class instance to start recording ram usage.
    Initial ram usage is substracted to normalise the results.
    """

    path = '/home/xoel/Escritorio/pyMKT/rebase/'
    mem_f = 'memory.log'
    state_f = 'state.log'
    command_dft = "cd "+path+"; sh memorizer.sh &"


    def __init__(self, command=command_dft):
        if bool(os.system(command)):
            raise(RuntimeError)


    def stop(self):
        with open(self.path+self.state_f, 'wt') as f:
            f.write('STOP')
        
        return self.get()
    
    def get(self):

        with open(self.path+self.mem_f, 'rt') as f:
            n = []
            f.readline()
            while True:
                l = f.readline()
                if not l: break
                n.append(int(l)*1024)
        self.results = np.array(n)

        return self.results
