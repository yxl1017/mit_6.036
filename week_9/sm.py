from util import *

class SM:
    start_state = None  # default start state

    def transition_fn(self, s, x):
        '''s:       the current state
           x:       the given input
           returns: the next state'''
        raise NotImplementedError

    def output_fn(self, s):
        '''s:       the current state
           returns: the corresponding output'''
        raise NotImplementedError

    def transduce(self, input_seq):
        '''input_seq: the given list of inputs
           returns:   list of outputs given the inputs'''
        # Your code here
        old_s = self.start_state
        output_seq = []
        for x in input_seq:
            #print('old_s =', old_s)
            new_s = self.transition_fn(old_s, x)
            #print('new_s =', new_s)
            output_seq.append(self.output_fn(new_s))
            #print('s_seq =', s_seq)
            old_s = new_s
        return output_seq
            


class Accumulator(SM):
    start_state = 0

    def transition_fn(self, s, x):
        return s + x

    def output_fn(self, s):
        return s


class Binary_Addition(SM):
    start_state = (0, 0) # (carry, digit)

    def transition_fn(self, s, x):
        carry, digit = s
        b0, b1 = x
        total = b0 + b1 +carry
        if total > 1:
            return (1, total%2)
        else:
            return (0, total%2)

    def output_fn(self, s):
        carry, digit = s
        return digit


class Reverser(SM):
    start_state = None
    end = False
    sequence1 = []
    def transition_fn(self, s, x):
        # Your code here
        if x == 'end':
            self.end = True
            
        if not self.end:
            self.sequence1.append(x)
            return None
        else:
            try:
                return self.sequence1.pop()
            except:
                return None


    def output_fn(self, s):
        return s


class RNN(SM):
    def __init__(self, Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2):
        # Your code here
        self.Wsx = Wsx
        self.Wss = Wss
        self.Wo = Wo
        self.Wss_0 = Wss_0
        self.Wo_0 = Wo_0
        self.f1 = f1
        self.f2 = f2
        self.start_state = np.zeros((self.Wss.shape[0], 1))
        
    def transition_fn(self, s, x):
        return self.f1(np.dot(self.Wss, s) + np.dot(self.Wsx, x) + self.Wss_0)
    def output_fn(self, s):
        return self.f2(np.dot(self.Wo, s) + self.Wo_0)

# 1.5) Accumulator Sign RNN
Wsx =    np.array([[1]])
Wss =    np.array([[1]])
Wo =     np.array([[1000]])
Wss_0 =  np.array([[0]])
Wo_0 =   np.array([[0]])
f1 =     lambda x: x
f2 =     np.tanh
acc_sign_rnn = RNN(Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2)

# 1.6) Autoregressive RNN
Wsx =    np.array([[1], [0], [0]]).reshape(-1, 1)
Wss =    np.array([0, 0, 0, 1, 0, 0, 0, 1, 0]).reshape(3, 3)
Wo =     np.array([1, -2, 3]).reshape(1, -1)
Wss_0 =  np.array([[0]])
Wo_0 =   np.array([[0]])
f1 =    lambda x: x # Your code here
f2 =    lambda x: x # Your code here
auto_rnn = RNN(Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2)