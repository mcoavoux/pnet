
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
        
        for p in self.model.parameters_list():
            p.scale_gradient(0)
        
        return loss
