import torch

class SignSTFunction(torch.autograd.Function):
    ''' Straight-Through (w/ saturation) for Sign Function.
    '''
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        input = input.sign()
        return input

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0

        grad_input = grad_input * (1 - input.tanh() ** 2)

        return grad_input

sign = SignSTFunction.apply
