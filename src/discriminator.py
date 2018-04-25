
import _dynet as dy

from classifier import MLP_sigmoid


class Discriminator:
    def __init__(self, input_size, output_size, hidden_layers, dim_hidden, activation, model, trainer):
        self.model = model
        self.mlp = MLP_sigmoid(input_size, output_size, hidden_layers, dim_hidden, dy.rectify, self.model)
        self.trainer = trainer
    
    
    def train_real(self, input, target):
        # train the discriminator to retrieve information
        # update only the discriminator parameters
        loss = self.mlp.get_loss(input, target)
        loss.backward()
        self.trainer.update()
        
        return loss
    
    def train_fake(self, input, fake_target):
        # Fool the discriminator
        # update only the main model parameters
        
        loss = self.mlp.get_loss(input, fake_target)
        
        return loss
    
    def zero_gradient(self):
        for p in self.mlp.get_parameter_list():
            p.scale_gradient(0)



class Generator:
    def __init__(self, args, vocabulary, model, trainer):
        self.vocabulary = vocabulary
        self.dim_lstm = args.dim_wrnn
        self.dim_embeddings = 32
        
        self.model = model
        self.trainer = trainer
        
        if "char" in args.generator:
            self.voc_size = vocabulary.size_chars()
        else:
            self.voc_size = vocabulary.size_words()
        
        self.lu = model.add_lookup_parameters((self.voc_size, self.dim_embeddings))
        self.lstm = dy.LSTMBuilder(1, self.dim_embeddings, self.dim_lstm, model)
        
        self.h2o = model.add_parameters((self.voc_size, self.dim_lstm))
        self.b  = model.add_parameters((self.voc_size))
    
    def train_real(self, input, targets, epsilon = 1e-10):
        init_states = [input, dy.zeros(self.dim_lstm)]
        
        state = self.lstm.initial_state(init_states)

        loss = dy.zeros(1)
        W = dy.parameter(self.h2o)
        b = dy.parameter(self.b)
        
        state = state.add_input(self.lu[targets[0]])
        
        for target in targets[1:]:
            #print(type(W), type(state.output()))
            loss += dy.pickneglogsoftmax(W * state.output() + b + epsilon, target)
            
            embedding = self.lu[target]
            state = state.add_input(embedding)
        
        loss.backward()
        self.trainer.update()
        
        return loss

    def train_fake(self, input, targets, epsilon = 1e-10):
        init_states = [input, dy.zeros(self.dim_lstm)]
        
        state = self.lstm.initial_state(init_states)

        loss = dy.zeros(1)
        W = dy.parameter(self.h2o)
        b = dy.parameter(self.b)
        
        state = state.add_input(self.lu[targets[0]])
        
        for target in targets[1:]:
            loss += dy.pickneglogsoftmax(W * state.output() + b + epsilon, target)
            
            embedding = self.lu[target]
            state = state.add_input(embedding)
        
        return loss
    
    def zero_gradient(self):
        for p in [self.lu, self.h2o, self.b] + self.lstm.get_parameters()[0]:
            p.scale_gradient(0)


    def generate(self, init_states):
        init_states = [dy.nobackprop(s) for s in init_states]
        states = self.lstm.initial_state(init_states)
        result = [None]
        while result[-1] != vocabulary.STOP_I:
            unormalized_distribution = W * state.output() + b
            vec = unormalized_distribution.value()
            prediction = np.argmax(dis_vec)
            result.append(prediction)
            
            states.add_input(self.lu[prediction])
        
        return "".join(result)

    def size(self):
        return self._size

    def set_dropout(self, v):
        self.lstm.set_dropout(v)

    def disable_dropout(self):
        self.lstm.disable_dropout(v)
