import datetime
import os
import rospy

from absl import app
from absl import flags
import numpy as np
from ravens import agents
from ravens.dataset import Dataset
from ravens.tasks import cameras
from ravens.utils import utils
from ravens import tasks

from ravens.environments.environment import Environment

import tensorflow as tf
from matplotlib import pyplot as plt
import pickle
import time

import torch
from torch.utils import data
from torch.autograd import Variable, variable

import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from spatial_softmax import SpatialSoftmax

import pybullet as p

from skimage import draw, data

import bc_softmax2d_aux
import bc_softmax_aux


assets_root = "/home/robot/Downloads/ravens/ravens/environments/assets/"
dataset_root = "/data/ravens_demo/"
model_save_rootdir = "/data/trained_model/"
#task_name = "place-red-in-green"
task_name = "block-insertion-nofixture"
mode = 'train'
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
        #   C*H*W
        self.mlp_common = torch.nn.Linear(32*53*73,64).cuda()
        
        self.mlp2 = torch.nn.Linear(70,35).cuda()
        self.mlp3 = torch.nn.Linear(35,8).cuda()

        self.mlp_eeaux1 = torch.nn.Linear(64,32).cuda()
        self.mlp_eeaux2 = torch.nn.Linear(32,3).cuda()
        
        self.mlp_objaux1 = torch.nn.Linear(64,32).cuda()
        self.mlp_objaux2 = torch.nn.Linear(32,3).cuda()

        self.spatial_softmax_layer = SpatialSoftmax(53,73,32)
        self.softmax2d_layer = nn.Softmax2d()

    def visual_feature_predict(self, im):
        x = self.conv1(im)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        #spatial_softmax_result = self.spatial_softmax_layer.forward(x.contiguous())
        spatial_softmax_result = self.softmax2d_layer(x)
        flattened_im = spatial_softmax_result.contiguous().view(spatial_softmax_result.size(0),-1)
        flattened_im = F.relu(self.mlp_common(flattened_im))

        return flattened_im

    def aux_predict(self, im_feature):
        

        x = F.relu(self.mlp_eeaux1(im_feature))
        eeaux_result = self.mlp_eeaux2(x)

        x = F.relu(self.mlp_objaux1(im_feature))
        objaux_result = self.mlp_objaux2(x)
        
        return eeaux_result, objaux_result

    def action_predict(self, im_feature, aux_feature_ee, aux_feature_obj):
        x = torch.cat((im_feature,aux_feature_ee), 1)
        x = torch.cat((x,aux_feature_obj), 1)
        x = F.relu(self.mlp2(x))
        x = self.mlp3(x)
        return x

    



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
        self.opt = torch.optim.Adam(self.model.parameters(),lr = 0.0001)
        
        #self.loss_func = 0.7 * nn.L1Loss() + 0.3 * nn.MSELoss
        

    def load_pretrained_model(self,filename):
        self.model.load_state_dict(torch.load(filename))
        

    def act(self, obs, verbose = False, ret_aux = False):
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

        im_feature = self.model.visual_feature_predict(x_tensor)
        aux_feature_ee, aux_feature_obj = self.model.aux_predict(im_feature)
        aux_array_ee = aux_feature_ee.squeeze(0).cpu().detach().numpy()
        aux_array_obj = aux_feature_obj.squeeze(0).cpu().detach().numpy()

        output = self.model.action_predict(im_feature,aux_feature_ee, aux_feature_obj)

        
        action_array = output.squeeze(0).cpu().detach().numpy()
        p0 = tuple(action_array[0:3])
        p1 = tuple(action_array[3:7])
        if(action_array[7] > 0.5):
            g = 1
        else:
            g = 0
        
        action['pose'] = (p0,p1)
        action['grasp'] = g

        predicted_aux = {
            'predicted_ee_position': tuple(aux_array_ee),
            'predicted_obj_pose' :tuple(aux_array_obj)

        }
        if verbose:
            print(action)
        if ret_aux == True:
            return action, predicted_aux
            
        return action





        
    def train_full_model(self,dataset):
        n_episodes = dataset.n_episodes
        batch_size = 8
        num_batches = n_episodes // batch_size
        f = open(model_save_rootdir+ "loss.txt","w+")

        for epoch in range(EPOCH):
            total_cost = 0
            for k in range(num_batches):
                start, end = k * batch_size, (k + 1) * batch_size
                # x, y are lists that contain obss and actions 
                #x, y, aux = self.get_minibatch(dataset, start, batch_size)
                x, y, aux = self.get_cached_minibatch(dataset, start, batch_size)
                x_tensor = torch.Tensor().cuda()
                y_tensor = torch.Tensor().cuda()
                aux_tensor = torch.Tensor().cuda()
                
                for item_x in x:
                    tmp = torch.Tensor(item_x).unsqueeze(0).cuda()
                    x_tensor = torch.cat((x_tensor, tmp),0)
                for item_y in y:
                    tmp = torch.Tensor(item_y).unsqueeze(0).cuda()
                    y_tensor = torch.cat((y_tensor, tmp),0)
                for item_aux in aux:
                    tmp = torch.Tensor(item_aux).unsqueeze(0).cuda()
                    aux_tensor = torch.cat((aux_tensor, tmp),0)
                
                
                batch_x = Variable(x_tensor)
                batch_x = batch_x.permute(0,3,1,2)
                batch_y = Variable(y_tensor)
                batch_aux = Variable(aux_tensor)


                

                
                # generate visual feature point with cnn
                im_feature = self.model.visual_feature_predict(batch_x)


                # cutoff backward propogation to main network
                im_feature_aux = im_feature.detach()
                # predict ee_position and object pose 
                aux_feature = self.model.aux_predict(im_feature_aux)
                # generate action from visual feature and predicted aux states
                action = self.model.action_predict(im_feature,aux_feature)


                #计算误差
                aux_loss = F.mse_loss(aux_feature,batch_aux)
                
                full_loss = F.mse_loss(action,batch_y) + F.l1_loss(action,batch_y) + aux_loss
                #清空上一次梯度
                self.opt.zero_grad()
                #误差反向传递
                aux_loss.backward()
                full_loss.backward()
                #优化器参数更新
                self.opt.step()
                total_cost += full_loss.item()
            print("Epoch = {epoch}, cost = {le}".format(
                epoch = epoch , le = total_cost / num_batches))
            f.write("%f\n"%(total_cost))
            if epoch % 200 == 0:
                filename = "full_model_%d_epoch"%(epoch)
                
                torch.save(self.model.state_dict(), model_save_rootdir + filename)
                


    def aux_pretraining(self,dataset, load_model = None, batch_size = 8):
        #n_episodes = dataset.n_episodes
        
        num_batches = self.n_episodes // batch_size
        f = open(model_save_rootdir+ "pretrain_loss.txt","w+")
        # load pretrained model if available
        if load_model != None:
            self.load_pretrained_model(model_save_rootdir + load_model)
        
        for epoch in range(EPOCH):
            total_cost = 0
            for k in range(num_batches):
                start, end = k * batch_size, (k + 1) * batch_size
                # x, y are lists that contain obss and actions 
                #x, y, aux = self.get_minibatch(dataset, start, batch_size)
                x, y, aux = self.get_cached_minibatch(dataset, start, batch_size)
                x_tensor = torch.Tensor().cuda()
                y_tensor = torch.Tensor().cuda()
                eeaux_tensor = torch.Tensor().cuda()
                objaux_tensor = torch.Tensor().cuda()
                for item_x in x:
                    tmp = torch.Tensor(item_x).unsqueeze(0).cuda()
                    x_tensor = torch.cat((x_tensor, tmp),0)
                for item_y in y:
                    tmp = torch.Tensor(item_y).unsqueeze(0).cuda()
                    y_tensor = torch.cat((y_tensor, tmp),0)
                for item_aux in aux:
                    ee_position = item_aux[0:3]
                    obj_position = item_aux[3:6]

                    tmp = torch.Tensor(ee_position).unsqueeze(0).cuda()
                    eeaux_tensor = torch.cat((eeaux_tensor, tmp),0)

                    tmp = torch.Tensor(obj_position).unsqueeze(0).cuda()
                    objaux_tensor = torch.cat((objaux_tensor, tmp),0)
                
                
                batch_x = Variable(x_tensor)
                batch_x = batch_x.permute(0,3,1,2)
                batch_y = Variable(y_tensor)
                batch_eeaux = Variable(eeaux_tensor)
                batch_objaux = Variable(objaux_tensor)



                

                
                im_feature = self.model.visual_feature_predict(batch_x)
                aux_feature_ee, aux_feature_obj = self.model.aux_predict(im_feature)
                
                #计算误差
                #loss = self.loss_func(output, batch_y)
                loss = F.mse_loss(aux_feature_ee,batch_eeaux) + F.mse_loss(aux_feature_obj,batch_objaux) 
                
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
            if epoch % 200 == 0:
                filename = "pretrain_model_%d_epoch"%(epoch)
                
                torch.save(self.model.state_dict(), model_save_rootdir + filename)
                
    def train_model_with_pretrained_cnn(self,dataset):
        n_episodes = dataset.n_episodes
        batch_size = 8
        num_batches = n_episodes // batch_size
        f = open(model_save_rootdir+ "pretrain_loss.txt","w+")
        self.load_pretrained_model(model_save_rootdir + "pretrain_model_1000_epoch")

        for epoch in range(EPOCH):
            total_cost = 0
            for k in range(num_batches):
                start, end = k * batch_size, (k + 1) * batch_size
                # x, y are lists that contain obss and actions 
                #x, y, aux = self.get_minibatch(dataset, start, batch_size)
                x, y, aux = self.get_cached_minibatch(dataset, start, batch_size)
                x_tensor = torch.Tensor().cuda()
                y_tensor = torch.Tensor().cuda()
                aux_tensor = torch.Tensor().cuda()
                
                for item_x in x:
                    tmp = torch.Tensor(item_x).unsqueeze(0).cuda()
                    x_tensor = torch.cat((x_tensor, tmp),0)
                for item_y in y:
                    tmp = torch.Tensor(item_y).unsqueeze(0).cuda()
                    y_tensor = torch.cat((y_tensor, tmp),0)
                for item_aux in aux:
                    tmp = torch.Tensor(item_aux).unsqueeze(0).cuda()
                    aux_tensor = torch.cat((aux_tensor, tmp),0)
                
                
                batch_x = Variable(x_tensor)
                batch_x = batch_x.permute(0,3,1,2)
                batch_y = Variable(y_tensor)
                batch_aux = Variable(aux_tensor)


                

                # generate visual feature point with cnn
                im_feature = self.model.visual_feature_predict(batch_x)
                
                # predict ee_position and object pose 
                aux_feature = self.model.aux_predict(im_feature)
                aux_feature = aux_feature.detach()
                im_feature = im_feature.detach()
                # generate action from visual feature and predicted aux states
                action = self.model.action_predict(im_feature,aux_feature)


                #计算误差
                #loss = self.loss_func(output, batch_y)
                loss = F.mse_loss(action,batch_y) + F.l1_loss(action,batch_y)
                #清空上一次梯度
                self.opt.zero_grad()
                #误差反向传递
                loss.backward()
                #优化器参数更新
                self.opt.step()
                total_cost += loss.item()
            print("Epoch = {n_epoch}, cost = {le}".format(
                n_epoch = epoch + 1 , le = total_cost / num_batches))
            f.write("%f\n"%(total_cost))
            if (epoch + 1) % 200 == 0:
                filename = "model_%d_epoch"%(epoch + 1)
                
                torch.save(self.model.state_dict(), model_save_rootdir + filename)
            
            
    def load_whole_dataset(self, dataset, n_max_demos = 1):
        self.episodes = []
        
        n_episodes = dataset.n_episodes
        if n_max_demos <= n_episodes:
            self.n_episodes = n_max_demos
        else:
             self.n_episodes = n_episodes
        
        s_set = []
        a_set = []
        aux_set = []
        
        for i in range(self.n_episodes):
                  
            episode = dataset.get_episode(i)
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

                ee_position = np.array(step[4]['ee_pose'][0])
                obj_position = np.array(step[4]['obj_pose'][0])
                obj_orientation = np.array(step[4]['obj_pose'][1])

                obj_pose = np.concatenate((obj_position, obj_orientation))
                s_aux = np.append(ee_position, obj_pose)
                aux_set.append(s_aux)

            print("episode %d has been loaded"%i)

        self.s_set = s_set
        self.a_set = a_set
        self.aux_set = aux_set
        
        return s_set, a_set, aux_set
    
    def get_minibatch(self,dataset,start,batch_size = 1):
        s_set = []
        a_set = []
        aux_set = []
        
        for i in range(batch_size - 1):
            episode_id = start + i
            # all episodes have been loaded
            if episode_id > (dataset.n_episodes - 1):
                return None
            #episode = dataset.get_episode(episode_id)
            episode = self.episodes[episode_id]
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

                ee_position = np.array(step[4]['ee_pose'][0])
                obj_position = np.array(step[4]['obj_pose'][0])
                obj_orientation = np.array(step[4]['obj_pose'][1])

                obj_pose = np.concatenate((obj_position, obj_orientation))
                s_aux = np.append(ee_position, obj_pose)
                aux_set.append(s_aux)


        return s_set, a_set, aux_set

    def get_cached_minibatch(self,dataset, start,batch_size = 1):
        end = start + batch_size
        if end > (self.n_episodes ):
            end = self.n_episodes 
        s_set = self.s_set[start:end]
        a_set = self.a_set[start:end]
        aux_set = self.aux_set[start:end]
        return s_set, a_set, aux_set

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
    rospy.init_node('raven_vive_teleop')
    rate = rospy.Rate(60.0)
    train_dataset = Dataset(os.path.join(dataset_root, 'ravens_demo-1623923819265029807'), use_aux = True)
    max_demos = train_dataset.n_episodes

    episodes = np.random.choice(range(max_demos), max_demos, False)
    print(episodes)
    train_dataset.set(episodes)

    env = Environment(
      assets_root,
      disp=True,
      hz=60)
    
    task = tasks.names[task_name]()
    task.mode = mode

    
    env.set_task(task)
    obs = env.reset()
    info = None
    
    seed = 0


    agent = bc_softmax_aux.BC_Agent()
    #agent = bc_softmax2d_aux.BC_Agent()
    agent.load_pretrained_model(model_save_rootdir + "pretrain_model_4400_epoch")
    agent.act(obs)

    episode_steps = 0
    n_episode = 1
    aux= {
        'ee_pose': ((0,0,0),(0,0,0,1)),
        'obj_pose': ((0,0,0),(0,0,0,1))
    }



    plt.ion()
    im_color = obs['color'][0]
    img_draw  = plt.imshow(im_color, interpolation='nearest', cmap ='gray')
    

    while not rospy.is_shutdown():
        (obs, act, _, _,aux_dataset), _ = train_dataset.sample()
        
        im_color = obs['color'][0]
        img_draw.set_data(im_color)
        plt.pause(0.00000001)
       
        action, predicted_aux = agent.act(obs, ret_aux= True)
        #if action != None:
        #    print(action)
        #obj_pose = predicted_aux['predicted_obj_pose']
        obj_pose = predicted_aux['predicted_ee_position']
        #obj_pose = aux['ee_pose']
        marker_head_point = [obj_pose[0], obj_pose[1], obj_pose[2]]
        marker_tail_point = [obj_pose[0], obj_pose[1], obj_pose[2]+0.01]
        p.addUserDebugLine(marker_head_point,
                            marker_tail_point,
                            lineColorRGB=[0, 1, 0],
                            lifeTime=3,
                            lineWidth=5)
    
        
        obs, reward, done, info, aux = env.step_simple(action,use_aux = True, stand_still = True)
        #obs, reward, done, info, aux = env.step_simple(action,use_aux = True)
        plt.pause(3)
        print(aux_dataset)
        print(predicted_aux)
        auxg = np.array(aux_dataset['ee_pose'][0],dtype= float)
        auxp = np.array(predicted_aux['predicted_ee_position'],dtype= float)
        d = np.sum(np.square(auxg-auxp)) / 3
        print('mse:%f\n'%d)
        
        # im_depth = obs['depth'][0]
        # img_draw.set_data(im_depth)
        # plt.pause(0.00000001)
        # plt.show()
        # print( im_color.shape)

        # print(obs['color'])
        s = "Episode:%d, steps:%d"%(n_episode, episode_steps)
        print(s)
        if(done):
            n_episode += 1
            episode_steps = 0
       
          
            seed += 1
            
            episode = []

        else:
            episode_steps += 1

        if episode_steps > 100:
            episode = []
            episode_steps = 0
            n_episode += 1

            obs = env.reset()
     
     
        


        

        
        rate.sleep()
    


if __name__ == "__main__":
    # execute only if run as a script
    main()