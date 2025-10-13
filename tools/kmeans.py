import numpy as np
import os
from mmcv import track_parallel_progress
from sklearn.cluster import MiniBatchKMeans
import faiss
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from nuscenes.utils import splits
import tqdm
import torch
import random
from functools import partial
from pprint import pprint
import time

class SemanticClusters:
    def __init__(
        self, 
        num_classes=17, 
        n_clusters=50, 
        cluster_mode='faiss',
        num_query_3d=1600, 
        base_query_num=100,
        base_sample_num=5,
        num_worker=2,
        batch_size=1000,
        random_state=42,
        pc_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
        voxel_size=[0.4, 0.4, 0.4]
    ):
        self.camera_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        self.occ_names = [
            'others', 'barrier', 'bicycle', 'bus', 'car', 
            'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 
            'truck', 'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 
            'manmade', 'vegetation'
        ]
        self.num_classes = num_classes
        self.n_clusters = n_clusters
        self.cluster_mode = cluster_mode
        self.num_query_3d = num_query_3d
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.grid_size = [int((self.pc_range[3] - self.pc_range[0]) / self.voxel_size[0]), int((self.pc_range[4] - self.pc_range[1]) / self.voxel_size[1]), int((self.pc_range[5] - self.pc_range[2]) / self.voxel_size[2])]
        self.base_sample_num = base_sample_num
        self.base_query_num = base_query_num
        self.num_worker = num_worker
        self.batch_size = batch_size
        self.random_state = random_state
        
    def load_scene(self, sample_path):
        '''
            Load the semantics labels from the given sample path.
        '''
        sample_path = os.path.join(sample_path, 'labels.npz')
        data = np.load(sample_path)
        data_dict = {name: data[name] for name in data.files}
        semantics = data_dict['semantics']
        return semantics

    def filter_semantics(self, semantics):
        '''
            Cluster the coordinates of different semantics classes for each sample.
        '''
        result = {
            i: np.zeros((0, 3)) for i in range(self.num_classes)
        }
        world_coords = self.generate_world_coords()
        for class_id in range(self.num_classes):
            indices = np.ascontiguousarray(np.argwhere(semantics == class_id))
            world_coords_class = world_coords[indices[:, 0], indices[:, 1], indices[:, 2]]
            if world_coords_class.shape[0] < self.n_clusters:
                result[class_id] = world_coords_class
                continue
            if len(world_coords_class) < self.n_clusters * 39:
                kmeans = MiniBatchKMeans(self.n_clusters, batch_size=self.batch_size, random_state=self.random_state)
                kmeans.fit(world_coords_class)
                centers = kmeans.cluster_centers_.astype(np.float32)
                result[class_id] = centers
            else:
                kmeans = faiss.Kmeans(3, self.n_clusters, niter=20)
                kmeans.train(world_coords_class.astype(np.float32))
                centers = kmeans.centroids
                result[class_id] = centers
        return result

    def merge_cluster_results(self, all_semantics_results):
        '''
            Merge the Kmeans results of all samples.
        '''
        all_semantics_dict = {
            class_id : np.array([0, 3], dtype=np.int32) for class_id in range(self.num_classes)
        }
        for class_id in range(self.num_classes):
            all_semantics_dict[class_id] = np.concatenate([res[class_id] for res in all_semantics_results])
        return all_semantics_dict

    def get_num_clusters(self, semantics_list):
        '''
            According to the proportion of voxel number, compute the number of clusters for each class.
        '''
        voxel_num_dict = {
            class_id : 0 for class_id in range(self.num_classes)
        }
        for semantics in semantics_list:
            for class_id in range(self.num_classes):
                voxel_num_dict[class_id] += np.sum(semantics == class_id)
        
        proportion_dict = {
            class_id : voxel_num_dict[class_id] / sum(voxel_num_dict.values()) for class_id in range(self.num_classes)
        }
        num_clusters = {
            class_id : int(self.num_query_3d * proportion_dict[class_id]) for class_id in range(self.num_classes)
        }
        return num_clusters
    
    def cluster_all_priori_points(self, semantics_dict, num_clusters=None, save_dir=''):
        all_priori_points = {i: [] for i in range(self.num_classes)}

        for class_id in tqdm.tqdm(range(self.num_classes)):
            cluster_points = semantics_dict[class_id]
            num_cluster = min(num_clusters[class_id], len(cluster_points))
            if num_cluster == 0:
                print(f"Class {self.occ_names[class_id]} has no priori points!")
                continue
            if cluster_points.shape[0] < num_cluster * 39:
                kmeans = MiniBatchKMeans(num_cluster, batch_size=self.batch_size, random_state=self.random_state)
                kmeans.fit(cluster_points)
                centers = np.round(kmeans.cluster_centers_, 3).astype(np.float32)
            else:
                kmeans = faiss.Kmeans(3, num_cluster, niter=20)
                kmeans.train(cluster_points.astype(np.float32))
                centers = kmeans.centroids
            all_priori_points[class_id] = np.concatenate([centers, np.ones((centers.shape[0], 1)) * class_id], axis=1)
            
        save_name = f"./cluster-centers_{self.num_classes}-classes_{self.n_clusters}_clusters.pkl"
        save_path = os.path.join(save_dir, save_name)
        with open(save_path, "wb") as f:
            pickle.dump(all_priori_points, f)
        return all_priori_points
    
    def generate_world_coords(self):
        W, H, Z = self.grid_size
        x = torch.arange(0, W, dtype=torch.float32)
        x = (x + 0.5) / W * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        y = torch.arange(0, H, dtype=torch.float32)
        y = (y + 0.5) / H * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        z = torch.arange(0, Z, dtype=torch.float32)
        z = (z + 0.5) / Z * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]

        xx = x[:, None, None].expand(W, H, Z)
        yy = y[None, :, None].expand(W, H, Z)
        zz = z[None, None, :].expand(W, W, Z)
        world_coors = torch.stack([xx, yy, zz], dim=-1) # actual space
        
        return world_coors.numpy()
    
    def main(self, data_root, train_only=False, balance=False, save_dir='', vis=True):
        scene_names = os.listdir(data_root)
        if train_only:
            scene_names = [scene_name for scene_name in scene_names if scene_name in splits.train]
        print("Scene Num: {} ...".format(len(scene_names)))
        
        tasks = []
        for scene_name in scene_names:
            scene_path = os.path.join(data_root, scene_name)
            sample_list = os.listdir(scene_path)
            for _ in range(self.base_sample_num):
                sample_idx = random.randint(0, len(sample_list)-1)
                sample_path = os.path.join(scene_path, sample_list[sample_idx])
                tasks.append(sample_path)
        
        semantics_lst = track_parallel_progress(self.load_scene, tasks, nproc=self.num_worker)
        print("Loaded {} samples ... ".format(len(semantics_lst)))
        
        if balance:
            num_clusters_dict = self.get_num_clusters(semantics_lst)
        else:
            num_clusters_dict = {
                class_id : self.base_query_num for class_id in range(self.num_classes)
            }
        
        semantics_filter_results = track_parallel_progress(self.filter_semantics, semantics_lst, nproc=self.num_worker)
        
        print("Number of clusters for each class: ")
        pprint(num_clusters_dict)
        
        merged_semantics_results = self.merge_cluster_results(semantics_filter_results)
        
        all_priori_points = self.cluster_all_priori_points(merged_semantics_results, num_clusters_dict, save_dir)
        
        if vis:
            img_save_path = os.path.join(save_dir, 'cluster_centers_3d.png')
            self.visualize_3d_clusters(all_priori_points, sample_interval=5, save_name=img_save_path)

    def visualize_3d_clusters(self, class_clusters, sample_interval=5, save_name='cluster_centers_3d.png'):
        """
        3D visualization of cluster centers for each class.
        
        Parameters:
            class_clusters: Dictionary, keys are class IDs, values are lists of cluster center coordinates [(x,y,z), ...]
            save_name: Filename for saving the visualization image
        """
        fig = plt.figure(figsize=(14, 8))      
        ax = fig.add_subplot(111, projection='3d')
        plt.rcParams['font.size'] = 15
        ax.set_title('3D Visualization of Class Clusters')
        
        colors = plt.cm.get_cmap('tab20', self.num_classes)
        plt.rcParams['font.size'] = 10   
        for class_id, centers in class_clusters.items():
            if len(centers) == 0:
                print(f"Class {self.occ_names[class_id]} has no cluster centers!")
                continue
            
            centers_array = np.array(centers)
            centers_array = centers_array[::sample_interval]
            x, y, z = centers_array[:, 0], centers_array[:, 1], centers_array[:, 2]
            
            ax.scatter(
                x, y, z, 
                color=colors(class_id), 
                label=f'{self.occ_names[class_id]}',
                s=30,
                alpha=0.8,
                edgecolors='w'
            )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        ax.legend(
            bbox_to_anchor=(1, 1), 
            loc='upper left', 
            ncol=2
        )
        
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.savefig(save_name, dpi=300)

    def visualize_3d_clusters(self, class_clusters, sample_interval=5, save_name='cluster_centers_3d.png'):
        """
        3D visualization of cluster centers for each class.
        
        Parameters:
            class_clusters: Dictionary, keys are class IDs, values are lists of cluster center coordinates [(x,y,z), ...]
            save_name: Filename for saving the visualization image
        """
        fig = plt.figure(figsize=(14, 8))      
        ax = fig.add_subplot(111, projection='3d')
        plt.rcParams['font.size'] = 15
        ax.set_title('3D Visualization of Class Clusters')
        
        colors = plt.cm.get_cmap('tab20', self.num_classes)
        plt.rcParams['font.size'] = 10   
        for class_id, centers in class_clusters.items():
            if len(centers) == 0:
                print(f"Class {self.occ_names[class_id]} has no cluster centers!")
                continue
            
            centers_array = np.array(centers)
            centers_array = centers_array[::sample_interval]
            x, y, z = centers_array[:, 0], centers_array[:, 1], centers_array[:, 2]
            
            ax.scatter(
                x, y, z, 
                color=colors(class_id), 
                label=f'{self.occ_names[class_id]}',
                s=30,
                alpha=0.8,
                edgecolors='w'
            )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        ax.legend(
            bbox_to_anchor=(1, 1), 
            loc='upper left', 
            ncol=2
        )
        
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.savefig(save_name, dpi=300)
        
if __name__ == "__main__":
    sem_cluster = SemanticClusters(
        num_classes=17, 
        n_clusters=20, 
        num_query_3d=1600, 
        base_sample_num=5,
        base_query_num=300,
        num_worker=1
    )
    sem_cluster.main("./data/nuscenes/gts", train_only=True)
    
    # data_path = './cluster-centers_17-classes_5_clusters.pkl'
    # import pickle
    # with open(data_path, 'rb') as f:
    #     all_priori_points = pickle.load(f)
    # for class_id in range(17):
    #     cur_priori_points = {
    #         class_id: all_priori_points[class_id]
    #     }
    #     save_name = f'./clusters_results/cluster_centers_3d_class_{class_id}.png'
    #     sem_cluster.visualize_3d_clusters(cur_priori_points, sample_interval=1, save_name=save_name)
    