
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
        
        self.dim_out = dim_out
    
    def output_size(self):
        return self.dim_out

    def compute_output_layer(self, input):
        res = [input]
        for i, p in enumerate(self.parameters):
            W, b = dy.parameter(p[0]), dy.parameter(p[1])
            if i == len(self.parameters) - 1:
                res.append(dy.softmax(W * res[-1] + b))
            else:
                 res.append(self.activation(W * res[-1] + b))
        return res

    def get_loss(self, input, target, epsilon = 1e-10)):
        layers = self.compute_output_layer(input)
        return - dy.log(dy.pick(layers[-1], target) + epsilon)

    def get_prediction(self, input):
        layers = self.compute_output_layer(input)
        return np.argmax(layers[-1].value())
    
    def get_loss_and_prediction(self, input, target, epsilon = 1e-10)):
        layers = self.compute_output_layer(input)
        return - dy.log(dy.pick(layers[-1], target) + epsilon), np.argmax(layers[-1].value())


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


    def get_prediction(self, input):
        layers = self.compute_output_layer(input)
        output = layers[-1].value()
        res = {i for i, v in enumerate(output) if v > 0.5}
        return res


    def _get_loss_and_prediction(self, input, targets, epsilon = 1e-10):
        layers = self.compute_output_layer(input)
        output = layers[-1].value()
        res = {i for i in output if i > 0.5}
        
        log_out = dy.log(layers[-1] + epsilon)
        
        loss = dy.zeros(1)
        for t in targets:
            loss += dy.pick(log_out, t)
        
        r = np.random.randint(self.dim_out)
        while r in targets:
            r = np.random.randint(self.dim_out)
        loss += dy.log(1 - dy.pick(layers[-1], r) + epsilon)
        #loss -= dy.pick(log_out, r)
        
        return - loss, res


    def _get_loss(self, input, targets, epsilon = 1e-10):
        layers = self.compute_output_layer(input)
        
        log_out = dy.log(layers[-1] + epsilon)
        
        loss = dy.zeros(1)
        for t in targets:
            loss += dy.pick(log_out, t)
        
        r = np.random.randint(self.dim_out)
        while r in targets:
            r = np.random.randint(self.dim_out)
        loss += dy.log(1 - dy.pick(layers[-1], r) + epsilon)
        #loss -= dy.pick(log_out, r)

        return - loss


    def get_loss_and_prediction(self, input, targets, epsilon = 1e-10):
        layers = self.compute_output_layer(input)
        #dim = layers[-1].dim()[0][0]
        #ts = np.ones(dim)
        #for t in targets:
            #ts[t] = 0
        
        #e = dy.inputTensor(ts)
        #me = - e
        #last = dy.cmult(layers[-1], me) + e

        ys = dy.vecInput(self.dim_out)
        ys.set([1 if i in targets else 0 for i in range(self.dim_out)])
        loss = dy.binary_log_loss(layers[-1], ys)


        output = layers[-1].value()
        res = {i for i, v in enumerate(output) if v > 0.5}
        
        return dy.sum_elems(loss), res


    def get_loss(self, input, targets, epsilon = 1e-10):
        layers = self.compute_output_layer(input)
        
        #ts = np.ones(dim)
        #for t in targets:
            #ts[t] = 0
        
        #print()
        #print("output", layers[-1].value())
        #print(targets)
        #e = dy.inputTensor(ts)
        #me = - e
        #last = dy.cmult(layers[-1], me) + e
        
        #print("gradient", last.value())
        #log_loss = dy.log(last + epsilon)
        #print(log_loss.value())
        ys = dy.vecInput(self.dim_out)
        ys.set([1 if i in targets else 0 for i in range(self.dim_out)])
        loss = dy.binary_log_loss(layers[-1], ys)
        return dy.sum_elems(loss)


if __name__ == "__main__":
    import dynet
    
    
    model = dy.Model()
    trainer = dy.SimpleSGDTrainer(model)
    
    classifier = MLP_sigmoid(2, 2, 2, 10, dy.rectify, model)
    
    dataset = [([-1, -1], {0}),
               ([-1, 1], {1}),
               ([1, -1], {1}),
               ([1, 1], {0})]
    
    for e in range(10040):
        for xs, y in dataset:
            dy.renew_cg()
            x = dy.vecInput(2)
            x.set(xs)
            l = classifier.get_loss(x, y)
            l.backward()
            trainer.update()
        
        loss = 0
        acc = 0
        for xs, y in dataset:
            dy.renew_cg()
            x = dy.vecInput(2)
            x.set(xs)
            l,p = classifier.get_loss_and_prediction(x, y)
            if p == y:
                acc += 1
            loss += l.value()
        
        print("Epoch {} loss = {} acc = {:.2}".format(e, loss, acc / len(dataset)))

    
    

