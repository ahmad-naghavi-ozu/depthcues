import os
import torch
from torch.utils.data import Dataset, Sampler

from data.occlusion import OcclusionV4Dataset
from data.lightshadow import LightshadowV1Dataset
from data.perspective import PerspectiveV1Dataset
from data.size import SizeV2Dataset
from data.texturegrad import TexturegradV1Dataset
from data.elevation import ElevationV1Dataset

dataset_dict = {
    'occlusion': {'class': OcclusionV4Dataset, 'folder': 'occlusion_v4'}, 
    'lightshadow': {'class': LightshadowV1Dataset, 'folder': 'lightshadow_v1'}, 
    'perspective': {'class': PerspectiveV1Dataset, 'folder': 'perspective_v1'},
    'size': {'class': SizeV2Dataset, 'folder': 'size_v2'}, 
    'texturegrad': {'class': TexturegradV1Dataset, 'folder': 'texturegrad_v1'}, 
    'elevation': {'class': ElevationV1Dataset, 'folder': 'elevation_v1'}
}

class MultiCueDataset(Dataset):
    def __init__(
            self, 
            data_path, 
            cues=['occlusion', 'lightshadow', 'perspective', 'size', 'texturegrad', 'elevation'], 
            transform=None, 
            split='train', 
            return_path=False
    ):
        """
        Initialize the dataset with a list of datasets (one per task).
        Args:
            datasets (list of torch.utils.data.Dataset): A list of datasets, one for each task.
        """
        self.cues = cues
        self.split = split
        self.datasets = [
            dataset_dict[cue]['class'](os.path.join(data_path, dataset_dict[cue]['folder']), 
                                       transform=transform, split=split, return_path=return_path)
            for cue in cues
        ]
        self.num_tasks = len(self.datasets)
        self.samples_per_task = None
        if split != 'test':
            self.samples_per_task = min(len(dataset) for dataset in self.datasets)

    def __len__(self):
        """
        The length is defined as the number of samples from the smallest dataset 
        multiplied by the number of tasks to ensure balanced sampling.
        """
        if self.split == 'test':
            return sum(len(dataset) for dataset in self.datasets)
        return self.samples_per_task * self.num_tasks

    def __getitem__(self, indices):
        """
        Given a tuple (task_idx, sample_idx), return the appropriate sample from the 
        task-specific dataset.
        Args:
            indices (tuple): A tuple containing the task index and the sample index within the task.
        
        Returns:
            sample (tuple): A sample (data, label) from the appropriate task-specific dataset.
            task_idx (int): The index of the task from which the sample was drawn.
        """
        task_idx, sample_idx = indices
        cue = self.cues[task_idx]
        dataset = self.datasets[task_idx]  # Select the dataset corresponding to the task
        return dataset[sample_idx], cue  # Return the sample from the selected dataset


class MultiCueTrainValBatchSampler(Sampler):
    def __init__(self, dataset: MultiCueDataset, batch_size, steps_per_cue, drop_last=False, shuffle=False):
        """
        Custom batch sampler to sample batches from a single dataset (task) at a time.
        Args:
            datasets (list of torch.utils.data.Dataset): List of datasets (one per task).
            batch_size (int): Number of samples in each batch.
        """
        self.datasets = dataset.datasets
        self.num_tasks = dataset.num_tasks
        self.samples_per_task = dataset.samples_per_task
        self.batch_size = batch_size
        self.steps_per_cue = steps_per_cue
        self.drop_last = drop_last
        self.shuffle = shuffle

        # Calculate the number of full batches for the smallest dataset
        if drop_last:
            self.num_batches_per_task = self.samples_per_task // batch_size
        else:
            self.num_batches_per_task = (self.samples_per_task + batch_size - 1) // batch_size  # ceil division

    def __iter__(self):
        """
        Generate batches by alternating between tasks. Each batch contains samples
        from only one dataset at a time.
        """
        # Get shuffled indices for each task's dataset, i.e. select samples for each task
        all_task_indices = []
        for dataset in self.datasets:
            if self.shuffle:
                indices = torch.randperm(len(dataset)).tolist()
            else:
                indices = torch.arange(len(dataset)).tolist()
            indices = (indices * (self.samples_per_task // len(indices) + 1))[:self.samples_per_task]
            all_task_indices.append(indices)

        # Now we generate batches by alternating between tasks
        for batch_idx in range(0, self.num_batches_per_task, self.steps_per_cue):
            task_order = torch.randperm(self.num_tasks).tolist()
            for task_idx in task_order:  # Random task order each time
                indices = all_task_indices[task_idx]
                for step_idx in range(self.steps_per_cue):
                    cur_batch_idx = batch_idx + step_idx
                    if cur_batch_idx == self.num_batches_per_task:
                        break
                    batch_indices = indices[cur_batch_idx * self.batch_size: (cur_batch_idx + 1) * self.batch_size]

                    if self.drop_last and len(batch_indices) < self.batch_size:
                        continue

                    global_batch_indices = [(task_idx, idx) for idx in batch_indices]
                    yield global_batch_indices

    def __len__(self):
        """
        The length of the sampler is determined by the smallest dataset to ensure balanced training.
        """
        return self.num_batches_per_task * self.num_tasks 
    

class MultiCueTestBatchSampler(Sampler):
    def __init__(self, dataset: MultiCueDataset, batch_size):
        """
        Custom batch sampler to sample batches from a single dataset (task) at a time.
        Args:
            datasets (list of torch.utils.data.Dataset): List of datasets (one per task).
            batch_size (int): Number of samples in each batch.
        """
        self.datasets = dataset.datasets
        self.num_tasks = dataset.num_tasks
        self.batch_size = batch_size
        self.samples_per_task = [len(dataset) for dataset in self.datasets]
        self.num_batches_per_task = [(len(dataset) + self.batch_size - 1) // self.batch_size for dataset in self.datasets]

    def __iter__(self):
        """
        Generate batches by alternating between tasks. Each batch contains samples
        from only one dataset at a time.
        """
        # For each task, iterate all batches
        for task_idx in range(self.num_tasks):
            dataset = self.datasets[task_idx]

            # Get shuffled indices for the task's dataset
            indices = torch.arange(len(dataset)).tolist()

            # Generate batches for this task
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                global_batch_indices = [(task_idx, idx) for idx in batch_indices]
                yield global_batch_indices


    def __len__(self):
        """
        The length of the sampler is determined by the smallest dataset to ensure balanced training.
        """
        
        return sum(self.num_batches_per_task)  # ceil division
