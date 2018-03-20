
import _dynet as dy
import numpy as np


class MLP:
    def __init__(self, dim_in, dim_out, n_hidden, dim_hidden, activation, model):
        assert(n_hidden > 0)
        self.parameters = [(model.add_parameters((dim_hidden, dim_in)), 
                            model.add_parameters((dim_hidden,)))]
        for i in range(n_hidden-1):
            self.parameters.append((model.add_parameters((dim_hidden, dim_hidden)),
                                    model.add_parameters((dim_hidden))))
        self.parameters.append((model.add_parameters((dim_out, dim_hidden)),
                                model.add_parameters((dim_out))))
        self.activation = activation

    def compute_output_layer(self, input):
        res = [input]
        for i, p in enumerate(self.parameters):
            W, b = dy.parameter(p[0]), dy.parameter(p[1])
            if i == len(self.parameters) - 1:
                res.append(dy.softmax(W * res[-1] + b))
            else:
                 res.append(self.activation(W * res[-1] + b))
        return res

    def get_loss(self, input, target):
        layers = self.compute_output_layer(input)
        return - dy.log(dy.pick(layers[-1], target))

    def get_prediction(self, input):
        layers = self.compute_output_layer(input)
        return np.argmax(layers[-1].value())
    
    def get_loss_and_prediction(self, input, target):
        layers = self.compute_output_layer(input)
        return - dy.log(dy.pick(layers[-1], target)), np.argmax(layers[-1].value())


class MLP_sigmoid(MLP):
    def __init__(self, dim_in, dim_out, n_hidden, dim_hidden, activation, model):
        super().__init__(dim_in, dim_out, n_hidden, dim_hidden, activation, model)

    def compute_output_layer(self, input):
        res = [input]
        for i, p in enumerate(self.parameters):
            W, b = dy.parameter(p[0]), dy.parameter(p[1])
            if i == len(self.parameters) - 1:
                res.append(dy.logistic(W * res[-1] + b))
            else:
                 res.append(self.activation(W * res[-1] + b))
        return res

    def get_loss(self, input, targets):
        layers = self.compute_output_layer(input)
        
        dim = layers[-1].dim()[0][0]
        ts = np.ones(dim)
        for t in targets:
            ts[t] = 0
        
        e = dy.Expression(ts)
        me = - e
        last = dy.cmult(layers[-1], me) + e
        
        return - dy.esum(dy.log(last))

    def get_prediction(self, input):
        layers = self.compute_output_layer(input)
        output = layers[-1].value()
        res = {i for i in output if i > 0.5}
        return res
    
    def get_loss_and_prediction(self, input, targets):
        layers = self.compute_output_layer(input)
        dim = layers[-1].dim()[0][0]
        ts = np.ones(dim)
        for t in targets:
            ts[t] = 0
        
        e = dy.Expression(ts)
        me = - e
        last = dy.cmult(layers[-1], me) + e

        output = layers[-1].value()
        res = {i for i in output if i > 0.5}
        
        return - dy.esum(dy.log(last)), res






