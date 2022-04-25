import numpy as np
import copy
from PIL import Image
import random
from torch.utils.data import DataLoader



class SlidingTiles:
    def __init__(self, size=None, state=None, image=None):
        if state is None:
            if size is None:
                raise RuntimeError("Need a state size if not passing a state")
            state = np.reshape(range(size * size), (size, size))
        self.state = state
        empty = np.argwhere(self.state == 0)
        if len(empty) > 1:
            raise RuntimeError("State should only have one empty tile")
        self.empty = empty[0]
        self.size = len(state)  # Number of tiles along one axis, image is size * size (-1) tiles
        self.goal = np.reshape(range(self.size * self.size), (self.size, self.size))
        self.image = image
        if image is not None:
            self.load_image(image)
        else:
            self.generate_image()
    

    def reset(self, n=0):
        '''
        Resets the state to the goal state, if n is passed it will perform that many actions
        '''
        self.state = np.reshape(range(self.size * self.size), (self.size, self.size))
        empty = np.argwhere(self.state == 0)
        self.empty = empty[0]
        while (self.state == self.goal).all():
            self.move_random(n)

    
    def generate_vae_dataset(self, n_data, train_split, val_split, batch, usecuda):
        '''
        Generates the dataset for the vae, consisting of pairs of states and an action which takes place in between then
        '''
        data = []
        steps_since_valid = 0           # Prevents the dataset from storing more than one invalid action in a row
        while len(data) < n_data:
            move = np.random.randint(4)
            reward, _ = self.step(move)
            if reward == -2:
                # If the action didn't result in the state changing we only want to add it to the dataset once
                # Prevents having a large number of samples with invalid actions
                # But still allows the model to train on invalid examples
                steps_since_valid += 1
            else:
                steps_since_valid = 0
            prev = (np.expand_dims(self.render(), 0) / 127.5) - 1
            action = np.expand_dims(self.move_random(1)[0], 0)
            succ = (np.expand_dims(self.render(), 0) / 127.5) - 1
            if steps_since_valid < 2:
                data.append({'prev' : prev, 'succ' : succ, 'action' : action})
        random.shuffle(data)
        train_ind = int(len(data) * train_split)
        val_ind = int(len(data) * (train_split + val_split))
        train_dataloader = DataLoader(data[:train_ind], batch_size=batch, pin_memory=usecuda, drop_last=True)
        val_dataloader= DataLoader(data[train_ind:val_ind], batch_size=batch, pin_memory=usecuda, drop_last=True)
        test_dataloader = DataLoader(data[val_ind:], batch_size=batch, pin_memory=usecuda, drop_last=True)
        return {'train' : train_dataloader, 'val' : val_dataloader, 'test' : test_dataloader}


    def generate_image(self, width=15):
        '''
        Generates the grey tile image - chosen out of the thought that the network could learn from it easily
        '''
        t_w = width // self.size
        gradient = [i * (255 // (self.size * self.size)) for i in range(self.size * self.size)]
        image = []
        for tile in gradient:
            image.append(tile * np.ones((t_w, t_w)))
        self.tiled_image = np.array(image)
        self.t_w = t_w
        self.t_h = t_w
        self.image = "generated"
        

    def load_image(self, image, width=-1):
        '''
        Loads an image, e.g. passing "cat" will load the cat image from the images folder instead of the default grey tiles
        '''
        if width > 0:
            image = Image.open('images/' + image + '.jpg').resize((width, width)).convert('L')
        else:
            image = Image.open('images/' + image + '.jpg').convert('L')
        image = np.asarray(image)
        t_w = len(image) // self.size
        t_h = len(image[0]) // self.size
        self.tiled_image = np.array([image[i * t_w : (i + 1) * t_w, j * t_h : (j + 1) * t_h] for i in range(self.size) for j in range(self.size)])
        self.tiled_image[self.empty[0] * self.size + self.empty[1]] = 0
        self.t_w = t_w
        self.t_h = t_h


    def move_random(self, n):
        '''
        Chooses actions randomly and executes them in the environment, doesn't count unsuccessful moves
        (moves which don't result in the state changing)
        '''
        successful_moves = []
        while len(successful_moves) < n:
            move = np.random.randint(4)
            if self.step(move)[0] == -1:
                successful_moves.append(move)
        return successful_moves
        

    def step(self, move):
        '''
        Moves tile into empty square
        Returns the reward, which is -1 for a move that results in the state changing,
            -2 for a move that does not change the state, and 0 for a move that reaches the goal state.
            Also returns whether the goal state was reached, returns 1 if reached, otherwise 0
        '''
        next_empty = copy.copy(self.empty)
        valid = False

        if move == 0:
            # Move empty down
            if self.empty[0] + 1 < self.size:
                next_empty[0] += 1
                valid = True
        elif move == 1:
            # Move empty up
            if self.empty[0] - 1 > -1:
                next_empty[0] -= 1
                valid = True
        elif move == 2:
            # Move empty right
            if self.empty[1] + 1 < self.size:
                next_empty[1] += 1
                valid = True
        elif move == 3:
            # Move empty left
            if self.empty[1] - 1 > -1:
                next_empty[1] -= 1
                valid = True
        else:
            raise ValueError("Invalid move")
        
        self.state[self.empty[0], self.empty[1]] = self.state[next_empty[0], next_empty[1]]
        self.state[next_empty[0], next_empty[1]] = 0
        self.empty = next_empty

        done = 0
        if (self.state == self.goal).all():
            done = 1
            reward = 0
        else:
            reward = int(valid) - 2

        return reward, done
    

    def get_state(self):
        # Returns a copy of the state
        return copy.deepcopy(self.state)
    

    def render(self, normalize=False):
        '''
        Returns an array of the image displaying the current state
        '''
        if self.image == None:
            raise RuntimeError("Class not initialized with image")
        d = self.tiled_image[self.state.flatten(), :, :]
        r1 = np.reshape(d, (self.size, self.size, self.t_w, self.t_w))
        t1 = np.transpose(r1, (0, 2, 1, 3))
        img_arr = np.reshape(t1, (self.t_w * self.size, self.t_w * self.size))
        if normalize:
            # Normalizes between -1 and 1
            img_arr = (img_arr / 127.5) - 1
        return img_arr

    
    def save_image(self, save_name):
        '''
        Saves the current state as an image
        '''
        image = self.render()
        image = Image.fromarray(image).convert("L")
        image.save(save_name)
    

    def get_image(self):
        '''
        Returns a PIL image of the current state
        '''
        image = self.render()
        image = Image.fromarray(image).convert("L")
        return image