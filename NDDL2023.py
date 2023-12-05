import matplotlib.pyplot as plt
import numpy as np
import os 


class PC_layer:
    VALID_PARAMS = ['n', 'n_next', 'weights', 'inf_rate', 'learn_rate', 'activation', ]
    def __init__(self,n: int,n_next:int=0,*, inf_rate=0.05, learn_rate=0.005, activation="linear",):
        self.n = n
        self.n_next = n_next
        self.restart_activity()
        self.weights=[]
        if n_next is not None and n_next != 0:
            self.weights = self.generate_xavier_random_weights(self.n, self.n_next)
        self.inf_rate = np.float32(inf_rate) 
        self.learn_rate = np.float32(learn_rate)
        self.activation = activation
        return 
        
    @property
    def activation(self):
        return self._activation
    
    @activation.setter
    def activation(self,activation):
        self._activation = str(activation)
        self.set_activation()
        return
        
    def set_activation(self):
        activation = self.activation
        activation_functions = {
            "tanh": np.tanh,
            "relu": lambda x: np.maximum(x,0),
            "leaky relu": lambda x: np.maximum(x,0.1*x),
            "linear": lambda x: x,
            "sigmoid": lambda x: 1/(1 + np.exp(-x)), 
        }
        derivative_functions = {
            "tanh": lambda x: 1 - np.tanh(x)**2,
            "relu": lambda x: np.where(x > 0, 1, 0),
            "leaky relu": lambda x: np.where(x > 0, 1, 0.1),
            "linear": lambda x: np.ones_like(x) , 
            "sigmoid": lambda x: np.exp(-x)/(1+np.exp(-x))**2
        }
        if activation not in activation_functions:
            raise NotImplementedError(f"Invalid activation function: {activation}. Supported activations are: {', '.join(activation_functions.keys())}")

        self.activation_function = activation_functions[activation]
        self.act_derivative = derivative_functions[activation]
        return 
            
    def generate_xavier_random_weights(self, n_in:int, n_out:int, *, dist='uniform'):
        # using uniform rand:
        if dist == 'uniform':
            return (np.random.rand(n_in, n_out) - 0.5) * np.sqrt(2/(n_in + n_out))
        if dist == 'normal':
            return np.random.randn(n_in, n_out) * np.sqrt(2/(n_in + n_out))
        raise NotImplementedError(f"Unknown distribution: {dist}. Supported distributions are: 'uniform', 'normal'")
        
    def restart_activity(self):
        self.r = np.zeros([self.n], dtype = np.float32)
        self.e = np.zeros([self.n], dtype = np.float32)
        self.x = np.zeros([self.n], dtype = np.float32) 
     
    def inference_step_calc_error(self, r_next):
        '''calculates the error from next layer's prediction
        '''
        is_last_layer = self.n_next == 0
        if is_last_layer:
            return 
        # prediction based on the activity of next layer
        pred = self.weights.dot(r_next) 
        self.e = self.x - pred
                   
    def inference_step_calc_r(self, BU_error):
        is_first_layer = len(BU_error) == 0
        if is_first_layer:
            return
        df = self.act_derivative(self.x)
        dx = self.inf_rate * (df* sum(BU_error) - self.e)
        self.x += dx
        self.r = self.activation_function(self.x)
                
    def learning_step(self,  r_next):
        dW = np.outer(self.e, r_next)
        return dW
    
    def update_weights(self, dW):
        self.weights += self.learn_rate * dW
    
    def calculate_MSE(self): 
        MSE = sum(self.e**2)/ self.n
        return MSE
      
    def save(self, dir, name="layer"):
        os.makedirs(dir, exist_ok=True)
        file_path = os.path.join(dir, name) 
        try:        
            kwargs = {}
            for x in PC_layer.VALID_PARAMS:
                kwargs[x] = getattr(self, x)
            
            np.savez_compressed(file_path, **kwargs )
        except:
            raise Exception(f"Error saving layer at {file_path}")

    def load(self, dir, name = "layer.npz"):
        file_path = os.path.join(dir, name)
        try:
            data = np.load(file_path)
            for x in PC_layer.VALID_PARAMS:
                setattr(self,x,data[x]) 
        except:
            raise Exception(f"Error loading layer at {file_path}")

class Network():
    VALID_PARAMS = ['n_layers', 'architecture', 'n_iter_inference', 'inf_rate', 'learn_rate', 'activation']
    def __init__(self, architecture: list[int], *, n_iter_inference:int=40, inf_rate = 0.05, learn_rate = 0.005, activation = "linear", ):
        self.n_layers = len(architecture)
        self.architecture = architecture
        self.n_iter_inference = n_iter_inference
        self.layers: list[PC_layer] = []
        for i in range(self.n_layers-1): 
            self.layers.append(PC_layer(self.architecture[i],self.architecture[i+1]))
        self.layers.append(PC_layer(self.architecture[-1])) 
        self.inf_rate = inf_rate
        self.learn_rate = learn_rate
        self.activation = activation
        self.standardise_layer_params()
        
    def standardise_layer_params(self):
        for l in self.layers:
            l.inf_rate = self.inf_rate
            l.learn_rate = self.learn_rate
            l.activation = self.activation
        
    def reset_rates(self):
        for layer in self.layers:
            layer.restart_activity()  
    
    def inference_step(self, r_M = None, BU_error = []):
        for (l, layer) in enumerate(self.layers):
            if l<len(self.layers)-1:
                r_next = self.layers[l+1].r
            else:
                r_next = r_M
            layer.inference_step_calc_error( r_next)
            layer.inference_step_calc_r(BU_error = BU_error)
            if r_next is not None:
                BU_error = [layer.weights.T.dot(layer.e)]   
        return
    
    def infer(self, image, initialize =  True):
        if initialize:
            self.reset_rates()
            
        I = np.array(image).reshape([-1])
        self.layers[0].x = I 
        self.layers[0].r = self.activation(self.layers[0].x)
        # self.layers[0].r = I 
        
        for i in range(self.n_iter_inference):
            self.inference_step()         
        return

    def learning(self, r_M = None):
        for l, layer in enumerate(self.layers):
            if l<len(self.layers)-1:
                r_next = self.layers[l+1].r
            elif r_M is not None:
                r_next = r_M
            else:
                return
            dw=layer.learning_step(r_next = r_next)
            layer.update_weights(dw)
        return 
    
    def train(self, dataset , *, type='serial', epochs = 100, items_per_batch=100):
        if type == 'serial':
            self.train_serial(dataset, epochs)
        elif type == 'batch':
            self.train_batch(dataset, epochs, items_per_batch)
                 
    def train_serial(self, dataset , epochs = 100):
        for epoch in range(epochs):
            for image in dataset:
                self.infer(image)
                self.learning()    
                                
    def reconstruct(self, layer = 1, recon = None):
        if recon is None:
            recon = self.layers[layer].r
        for l in range(layer,0,-1):
            recon = self.layers[l-1].weights.dot(recon)
        return recon
    
    def calculate_MSE(self):
        e=[]
        for l in self.layers:
            e.append(l.calculate_MSE())
        return e
            
    def save(self,dir,name="model"):
        os.makedirs(dir, exist_ok=True)
        file_path = os.path.join(dir, name) 
        try:        
            kwargs = {}
            for x in Network.VALID_PARAMS:
                kwargs[x] = getattr(self, x)
            
            np.savez_compressed(file_path, **kwargs )
        except:
            raise Exception(f"Error saving network at {file_path}")
        
        for i in range(self.n_layers):
            self.layers[i].save(dir,name= name.rsplit(".",1)[0]+"layer"+str(i))
        print("saved: {}".format(file_path))

    def load(self,dir,name = "model.npz"):
        file_path = os.path.join(dir, name)
        try:
            data = np.load(file_path)
            for x in Network.VALID_PARAMS:
                setattr(self,x,data[x]) 
        except:
            raise Exception(f"Error loading layer at {file_path}")
        self.layers=[]
        for i in range(self.n_layers):
            L = PC_layer(1)
            L.load(dir,name= name.rsplit(".",1)[0]+"layer"+str(i)+".npz")
            self.layers.append(L)
        print("loaded: {}".format(file_path))
    
class MultimodalNetwork():
    VALID_PARAMS = ['n_iter_inference', 'inf_rate', 'learn_rate', 'activation']
    def __init__(self, mod1, mod2, joint,*, n_iter_inference:int=40, inf_rate = 0.05, learn_rate = 0.005, activation = "linear", **kwargs):
        self.mod1: Network = Network(mod1, n_iter_inference= n_iter_inference)
        self.mod2: Network = Network(mod2, n_iter_inference= n_iter_inference)
        self.joint: Network =  Network(joint, n_iter_inference= n_iter_inference)
        self.connect()
        self.n_iter_inference = n_iter_inference
        self.inf_rate = inf_rate
        self.learn_rate = learn_rate
        self.activation = activation
        self.standardise_networks()       
        
    def connect(self):
        L1 = self.mod1.layers[-1]
        L2 = self.mod2.layers[-1]
        L_joint = self.joint.layers[0]
        L1.n_next = L_joint.n
        L2.n_next = L_joint.n
        L1.weights = L1.generate_xavier_random_weights(L1.n, L1.n_next)
        L2.weights = L2.generate_xavier_random_weights(L2.n, L2.n_next)
    
    def standardise_networks(self):
        '''
        makes sure that all the networks and all the layers inside a network
        have the exact same parameters TODO: call this method inside the setter
        '''
        for net in [self.mod1,self.mod2,self.joint]:
            net.n_iter_inference = self.n_iter_inference
            net.inf_rate = self.inf_rate
            net.learn_rate = self.learn_rate
            net.activation = self.activation
            net.standardise_layer_params()

    def reset_rates(self):
        self.mod1.reset_rates()
        self.mod2.reset_rates()
        self.joint.reset_rates()
    
    def inference_step(self):
        #TODO: optimize
        # each modality performs a single inference step 
        self.mod1.inference_step(r_M = self.joint.layers[0].r)
        self.mod2.inference_step(r_M = self.joint.layers[0].r)
        # the joint modality performs a single inference 
        BU1 = self.mod1.layers[-1].weights.T.dot(self.mod1.layers[-1].e)
        BU2 = self.mod2.layers[-1].weights.T.dot(self.mod2.layers[-1].e)
        self.joint.inference_step(BU_error= [BU1, BU2])

    def infer(self,image1, image2, initialize = True):
        if initialize:
            self.reset_rates()
        self.mod1.layers[0].r = np.array(image1).reshape([-1])
        self.mod2.layers[0].r = np.array(image2).reshape([-1]) 
        for i in range(self.n_iter_inference):
            self.inference_step()         

    def learning(self):
        self.mod1.learning(r_M = self.joint.layers[0].r)
        self.mod2.learning(r_M = self.joint.layers[0].r)
        self.joint.learning()
        
    def train(self, data1, data2, epochs=30):
        for epoch in range(epochs):
            for (I1,I2) in zip(data1,data2):
                self.infer(I1,I2)
                self.learning()             
    
    def reconstruct(self,modal, layer):
        if modal == "uni":
            recon1 = self.mod1.reconstruct(layer)
            recon2 = self.mod2.reconstruct(layer)
        if modal=="joint":
            recon = self.joint.reconstruct(layer)
            rec1 = self.mod1.layers[-1].weights.dot(recon)
            rec2 = self.mod2.layers[-1].weights.dot(recon)
            recon1 = self.mod1.reconstruct(layer= self.mod1.n_layers-1,recon = rec1)
            recon2 = self.mod2.reconstruct(layer= self.mod2.n_layers-1,recon = rec2)
        return recon1, recon2
    
    def infer_unimodal(self,modality,image, initialize = True):
        '''
        presents the image to the specified modality,
        performs inference on one modality and the joint network
        '''
        if modality ==0:
            net = self.mod1
        else:
            net = self.mod2
            
        if initialize:
            self.reset_rates()
        net.layers[0].r = np.array(image).reshape([-1])
        for i in range(self.n_iter_inference):
            net.inference_step(r_M = self.joint.layers[0].r)
            #topmost layer gets no prediction and send no BU_error
            BU = net.layers[-1].weights.T.dot(net.layers[-1].e)
            self.joint.inference_step(BU_error= [BU])
    
    def calculate_MSE(self):
        e1 = self.mod1.calculate_MSE()
        e2 = self.mod2.calculate_MSE()
        e3 = self.joint.calculate_MSE()
        return e1,e2,e3
    
    def save(self,dir,name = "Multimodal",name1="Mod1.npz",name2="Mod2.npz"):
        os.makedirs(dir, exist_ok=True)
        file_path = os.path.join(dir, name) 
        try:        
            kwargs = {}
            for x in MultimodalNetwork.VALID_PARAMS:
                kwargs[x] = getattr(self, x)    
            np.savez_compressed(file_path, **kwargs )
        except:
            raise Exception(f"Error saving multimodal network at {file_path}")
        self.mod1.save(dir,name1)
        self.mod2.save(dir,name2)
        self.joint.save(dir,"joint")
        print("saved: {}".format(file_path))
    
    def load(self,dir, name, name1="Mod1.npz", name2="Mod2.npz"):
        file_path = os.path.join(dir, name)
        try:
            data = np.load(file_path)
            for x in MultimodalNetwork.VALID_PARAMS:
                setattr(self,x,data[x]) 
        except:
            raise Exception(f"Error loading layer at {file_path}")
        self.mod1.load(dir,name1)
        self.mod2.load(dir,name2)
        self.joint.load(dir,"joint")

     
    
