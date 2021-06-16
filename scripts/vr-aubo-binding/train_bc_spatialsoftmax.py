import datetime
import os

from absl import app
from absl import flags
import numpy as np
from ravens import agents
# from ravens.dataset import Dataset
from ravens.tasks import cameras
from ravens.utils import utils
import tensorflow as tf
from matplotlib import pyplot as plt
import pickle
import time

import torch
from torch.utils import data
from torch.autograd import Variable

import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from spatial_softmax import SpatialSoftmax

assets_root = "/home/robot/Downloads/ravens/ravens/environments/assets/"
dataset_root = "/data/ravens_demo/"
model_save_rootdir = "/data/trained_model/"
#task_name = "place-red-in-green"
task_name = "block-insertion-nofixture"
EPOCH = 3000

class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=4,
                            out_channels=80,
                            kernel_size=7,
                            stride=2
                            ),
            #torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
            #torch.nn.MaxPool2d(kernel_size = 2)
        ).cuda()
        #self.conv1 = self.conv1.cuda()
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=80,
                            out_channels=32,
                            kernel_size=1
                            ),

            torch.nn.ReLU()
        ).cuda()
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,
                            out_channels=32,
                            kernel_size=3
                            ),

            torch.nn.ReLU()
        ).cuda()
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,
                            out_channels=32,
                            kernel_size=3
                            ),

            torch.nn.ReLU()
        ).cuda()
        #self.mlp1 = torch.nn.Linear(73*53*32,100).cuda()
        self.mlp2 = torch.nn.Linear(64,50).cuda()
        self.mlp3 = torch.nn.Linear(50,8).cuda()

        self.spatial_softmax_layer = SpatialSoftmax(53,73,32)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        spatial_softmax_result = self.spatial_softmax_layer.forward(x.contiguous())
        
        #x = self.mlp1(x.contiguous().view(x.size(0),-1))
        x = self.mlp2(spatial_softmax_result)
        x = self.mlp3(x)
        return x


class Dataset:
    def __init__(self, path):
        self.path = path
        self.sample_set = []
        self.max_seed = -1
        self.n_episodes = 0

        # Track existing dataset if it exists.
        color_path = os.path.join(self.path, 'action')
        if tf.io.gfile.exists(color_path):
            for fname in sorted(tf.io.gfile.listdir(color_path)):
                if '.pkl' in fname:
                    seed = int(fname[(fname.find('-') + 1):-4])
                    self.n_episodes += 1
                    self.max_seed = max(self.max_seed, seed)

        self._cache = {}

    

    def set(self, episodes):
    
        self.sample_set = episodes

    def load(self, episode_id, images=True, cache=False):
    
        def load_field(episode_id, field, fname):

            # Check if sample is in cache.
            if cache:
                if episode_id in self._cache:
                    if field in self._cache[episode_id]:
                        return self._cache[episode_id][field]
                else:
                    self._cache[episode_id] = {}

            # Load sample from files.
            path = os.path.join(self.path, field)
            data = pickle.load(open(os.path.join(path, fname), 'rb'))
            if cache:
                self._cache[episode_id][field] = data
            return data

        # Get filename and random seed used to initialize episode.
        seed = None
        path = os.path.join(self.path, 'action')
        for fname in sorted(tf.io.gfile.listdir(path)):
            if f'{episode_id:06d}' in fname:
                seed = int(fname[(fname.find('-') + 1):-4])

                # Load data.
                color = load_field(episode_id, 'color', fname)
                depth = load_field(episode_id, 'depth', fname)
                action = load_field(episode_id, 'action', fname)
                reward = load_field(episode_id, 'reward', fname)
                info = load_field(episode_id, 'info', fname)

                # Reconstruct episode.
                episode = []
                for i in range(len(action)):
                    obs = {'color': color[i], 'depth': depth[i]} if images else {}
                    episode.append((obs, action[i], reward[i], info[i]))
                return episode, seed

    def sample(self, images=True, cache=False):


        # Choose random episode.
        if len(self.sample_set) > 0:  # pylint: disable=g-explicit-length-test
            episode_id = np.random.choice(self.sample_set)
        else:
            episode_id = np.random.choice(range(self.n_episodes))
        episode, _ = self.load(episode_id, images, cache)

        # Return random observation action pair (and goal) from episode.
        i = np.random.choice(range(len(episode) - 1))
        sample, goal = episode[i], episode[-1]
        
        return sample, goal
    
    def get_episode(self, episode_id, images=True, cache=False):


        
        episode, _ = self.load(episode_id, images, cache)

        return episode


class BC_Agent:
    def __init__(self) -> None:

        self.total_steps = 0
        self.crop_size = 64
        
        self.pix_size = 0.003125
        self.in_shape = (320, 160, 6)
        self.cam_config = cameras.D415_infront.CONFIG
        #self.models_dir = os.path.join(root_dir, 'checkpoints', self.name)
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])
        self.model = CNNnet()
        print(self.model)
        self.opt = torch.optim.Adam(self.model.parameters(),lr = 0.001)
        
        #self.loss_func = 0.7 * nn.L1Loss() + 0.3 * nn.MSELoss
        

    def load_pretrained_model(self,filename):
        self.model.load_state_dict(torch.load(filename))
        

    def act(self, obs, verbose = False):
        action = {}
        #resize picture to 160*120
        color_array = obs['color'][0]
        img = Image.fromarray(color_array).resize((160,120))
        #img = img.astype(np.float32) / 255.
        dep_array = obs['depth'][0]
        img_depth = Image.fromarray(dep_array).resize((160,120))
        # get array from resized image and normalization
        color_array = np.asanyarray(img) / 255.0
        dep_array = np.asanyarray(img_depth)
        #expand depth map shape to (120*160*1)
        tmp = np.expand_dims(dep_array,2)
        # concatenate color and depth to (120*160*4)
        rgbd_array = np.concatenate((color_array, tmp),2)

        #x_tensor = torch.Tensor().cuda()
        x_tensor = Variable(torch.Tensor(rgbd_array).unsqueeze(0).cuda())
        x_tensor = x_tensor.permute(0,3,1,2)
        output = self.model.forward(x_tensor)
        action_array = output.squeeze(0).cpu().detach().numpy()
        p0 = tuple(action_array[0:3])
        p1 = tuple(action_array[3:7])
        if(action_array[7] > 0.5):
            g = 1
        else:
            g = 0
        
        action['pose'] = (p0,p1)
        action['grasp'] = g
        if verbose:
            print(action)
        return action





        
    def train_model(self,dataset):
        n_episodes = dataset.n_episodes
        batch_size = 8
        num_batches = n_episodes // batch_size
        f = open(model_save_rootdir+ "loss.txt","w+")

        for epoch in range(EPOCH):
            total_cost = 0
            for k in range(num_batches):
                start, end = k * batch_size, (k + 1) * batch_size
                # x, y are lists that contain obss and actions 
                x, y = self.get_minibatch(dataset, start, batch_size)
                x_tensor = torch.Tensor().cuda()
                y_tensor = torch.Tensor().cuda()
                for item_x in x:
                    tmp = torch.Tensor(item_x).unsqueeze(0).cuda()
                    x_tensor = torch.cat((x_tensor, tmp),0)
                for item_y in y:
                    tmp = torch.Tensor(item_y).unsqueeze(0).cuda()
                    y_tensor = torch.cat((y_tensor, tmp),0)
                
                
                batch_x = Variable(x_tensor)
                batch_x = batch_x.permute(0,3,1,2)
                batch_y = Variable(y_tensor)

                

                #输入训练数据
                output = self.model.forward(batch_x)
                #计算误差
                #loss = self.loss_func(output, batch_y)
                loss = F.mse_loss(output,batch_y) + F.l1_loss(output,batch_y)
                #清空上一次梯度
                self.opt.zero_grad()
                #误差反向传递
                loss.backward()
                #优化器参数更新
                self.opt.step()
                total_cost += loss.item()
            print("Epoch = {epoch}, cost = {le}".format(
                epoch = epoch , le = total_cost / num_batches))
            f.write("%f\n"%(total_cost))
            if epoch % 100 == 0:
                filename = "model_%d_epoch"%(epoch)
                
                torch.save(self.model.state_dict(), model_save_rootdir + filename)
                




            

    def get_minibatch(self,dataset,start,batch_size = 1):
        s_set = []
        a_set = []
        
        for i in range(batch_size - 1):
            episode_id = start + i
            # all episodes have been loaded
            if episode_id > (dataset.n_episodes - 1):
                return None
            episode = dataset.get_episode(episode_id)
            episode_size = len(episode)
            for step in episode:
                #resize picture to 160*120
                color_array = step[0]['color'][0]
                img = Image.fromarray(color_array).resize((160,120))
                #img = img.astype(np.float32) / 255.
                dep_array = step[0]['depth'][0]
                img_depth = Image.fromarray(dep_array).resize((160,120))
                # get array from resized image and normalization
                color_array = np.asanyarray(img) / 255.0
                dep_array = np.asanyarray(img_depth)
                #expand depth map shape to (120*160*1)
                tmp = np.expand_dims(dep_array,2)
                # concatenate color and depth to (120*160*4)
                rgbd_array = np.concatenate((color_array, tmp),2)
                s_set.append(rgbd_array)
                p0 = np.array(step[1]['pose'][0])
                p1 = np.array(step[1]['pose'][1])
                g =  np.array(step[1]['grasp'])
                p = np.concatenate((p0, p1))
                action = np.append(p,g)
                a_set.append(action)
        return s_set, a_set

            

    def get_sample(self, dataset, augment=True):


        (obs, act, _, _), _ = dataset.sample()
        cmap, hmap = self.get_image(obs)

        # Get training labels from data sample.
        if act != None:
            pose = act['pose']
            grasp = act['grasp']
        else:
            pose = ((0.0, 0.0, 0.0),(0.0, 0.0, 0.0, 1.0))
            grasp = 0
      

        return cmap, hmap, pose, grasp

    def get_image(self, obs):
        """Stack color and height images image."""

        # if self.use_goal_image:
        #   colormap_g, heightmap_g = utils.get_fused_heightmap(goal, configs)
        #   goal_image = self.concatenate_c_h(colormap_g, heightmap_g)
        #   input_image = np.concatenate((input_image, goal_image), axis=2)
        #   assert input_image.shape[2] == 12, input_image.shape

        # Get color and height maps from RGB-D images.
        color = obs['color'][0]
        depth = obs['depth'][0]
        
        return color, depth


def main():

    train_dataset = Dataset(os.path.join(dataset_root, 'ravens_demo-1623740955852015910'))
    max_demos = train_dataset.n_episodes

    episodes = np.random.choice(range(max_demos), max_demos, False)
    print(episodes)
    train_dataset.set(episodes)

    agent = BC_Agent() 
    if hasattr(torch.cuda, 'empty_cache'):
	    torch.cuda.empty_cache()
    agent.train_model(train_dataset)

    
    #cmap, _, _, _  = agent.get_sample(train_dataset)

    plt.ion()
    
    img_draw = plt.imshow(cmap, interpolation='nearest')

    while True:
        cmap, hmap, pose, grasp  = agent.get_sample(train_dataset)
        img_draw.set_data(cmap)
        if grasp == 1:
            plt.pause(5)
        else:    
            plt.pause(0.0001)
        print('pose:{}, grasp:{}'.format(pose,grasp))
        time.sleep(1)
    


if __name__ == "__main__":
    # execute only if run as a script
    main()