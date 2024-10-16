import os
import shutil
from typing import Any, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import yaml

class DataMethod:
    def __init__(self, dict: Dict = None):
        self.dict = dict

    def __generatedata__(self, 
            **kwargs
            ) -> Any: 
        """
        This function is used to generate the synthetic data.
    
        Parameters:
            seq_len (int): The length of the sequence. Default is 100.
            dim (int): The dimension of the data. Default is 10.
            noise_scale (float): The scale of the noise. Default is 0.1.
    
        Returns:
            Tuple: The synthetic data.
        """

        seq_len = self.dict.get("seq_length", 100)
        dim = self.dict.get("dimension", 10)
        # Generate the data.
        x = torch.randn(seq_len, dim)
        return x

    def __transform__(self, 
            x: Any, 
            **kwargs
            ) -> Any:
        """
        This function is used to transform the data for training, validation and testing.
    
        Parameters:
            x (torch.Tensor): The data with shape (batch_size, seq_len, dim)
    
        Returns:
            x (torch.Tensor): The original data removing the last element with shape (batch_size, seq_len - 1, dim).
            y (torch.Tensor): The target data with shape (batch_size, seq_len - 1, dim).
        """
        y = x[..., 1:, :].clone()
        x = x[..., :-1, :]
        return x, y



class SyntheticDataGenerator: 
    def __init__(self, 
                    data_method: DataMethod,
                    data_dir: str = None, 
                    train_sample_size: int = 1000, 
                    val_sample_size: int = 100, 
                    test_sample_size: int = 100, 
                    num_sample_per_file_train: int = 20,
                    num_sample_per_file_val: int = 20,
                    num_sample_per_file_test: int = 20,
                    **kwargs
                 ):
        """
        This class is used to generate synthetic data for testing purposes. 

        Parameters:
            data_dir (str): The directory where the data files are stored. Default is None.
            train_sample_size (int): The size of the training sequence. Default is 1000.
            val_sample_size (int): The size of the validation sequence. Default is 100.
            test_sample_size (int): The size of the test sequence. Default is 100.
            num_sample_per_file (int): The number of sequences per file. Default is 10.
            DataMethod_dict (Dict): The parameters for the generating function. Default is None.
        """

        self.data_dir = data_dir
        # check if the data directory exists
        if not os.path.exists(data_dir):
            pass 
        else:
            # delete the existing data directory
            shutil.rmtree(data_dir)
        os.makedirs(data_dir)
        os.makedirs(os.path.join(data_dir, "train"))
        os.makedirs(os.path.join(data_dir, "val"))
        os.makedirs(os.path.join(data_dir, "test"))
            
        self.train_sample_size = train_sample_size
        self.val_sample_size = val_sample_size
        self.test_sample_size = test_sample_size
        self.num_sample_per_file_train = num_sample_per_file_train
        self.num_sample_per_file_val = num_sample_per_file_val
        self.num_sample_per_file_test = num_sample_per_file_test
        self.data_method = data_method
        self.data_method_args_dict = kwargs

    def generate_data(self, 
                      return_index_to_file_dict: bool = False,
                      ) -> Tuple:
        """
        This function is used to generate the synthetic data (train, val, test) and save them to the data directory.

        Returns:
            Tuple: The index_to_file dictionary.
        """
        
        # create the index_to_file dictionary
        self.index_to_file_dict = {"data_dir": self.data_dir}

        # Generate the training data file by file.

        # generate the first sample
        train_data = []

        file_idx = 0
        for i in range(self.train_sample_size):
            train_data.append(self.data_method.__generatedata__(**self.data_method_args_dict))
            if (i + 1) % self.num_sample_per_file_train == 0:
                # torch save
                torch.save(torch.stack(train_data), os.path.join(self.data_dir, "train", f"{file_idx}.pt"))
                train_data = []
                file_idx += 1
        # save the remaining samples
        if (i+1) % self.num_sample_per_file_train != 0:
            torch.save(torch.stack(train_data), os.path.join(self.data_dir, "train", f"{file_idx}.pt"))
            file_idx += 1
        
        # update the index_to_file dictionary
        self.index_to_file_dict["train"] = {
                i: os.path.join(f"{i}.pt") for i in range(file_idx)
            }
        
        
        # add the train_sample_size, num_sample_per_file_train and num_files to the index_to_file_dict
        self.index_to_file_dict["train"]["data_dir"] = os.path.join(self.data_dir, "train")
        self.index_to_file_dict["train"]["sample_size"] = self.train_sample_size
        self.index_to_file_dict["train"]["num_sample_per_file"] = self.num_sample_per_file_train
        self.index_to_file_dict["train"]["num_files"] = file_idx

        # Generate the validation data file by file.
        val_data = []

        file_idx = 0
        for i in range(self.val_sample_size):
            val_data.append(self.data_method.__generatedata__(**self.data_method_args_dict))
            if (i + 1) % self.num_sample_per_file_val == 0:
                # torch save
                torch.save(torch.stack(val_data), os.path.join(self.data_dir, "val", f"{file_idx}.pt"))
                val_data = []
                file_idx += 1
        # save the remaining samples
        if (i+1) % self.num_sample_per_file_val != 0:
            torch.save(torch.stack(val_data), os.path.join(self.data_dir, "val", f"{file_idx}.pt"))
            file_idx += 1

        # update the index_to_file dictionary
        self.index_to_file_dict["val"] = {
            i: os.path.join(f"{i}.pt") for i in range(file_idx)
        }

        # add the val_sample_size, num_sample_per_file_val and num_files to the index_to_file_dict
        self.index_to_file_dict["val"]["data_dir"] = os.path.join(self.data_dir, "val")
        self.index_to_file_dict["val"]["sample_size"] = self.val_sample_size
        self.index_to_file_dict["val"]["num_sample_per_file"] = self.num_sample_per_file_val
        self.index_to_file_dict["val"]["num_files"] = file_idx

        # Generate the test data file by file.
        test_data = []

        file_idx = 0
        for i in range(self.test_sample_size):
            test_data.append(self.data_method.__generatedata__(**self.data_method_args_dict))
            if (i + 1) % self.num_sample_per_file_test == 0:
                # torch save
                torch.save(torch.stack(test_data), os.path.join(self.data_dir, "test", f"{file_idx}.pt"))
                test_data = []
                file_idx += 1
        # save the remaining samples
        if (i+1) % self.num_sample_per_file_test != 0:
            torch.save(torch.stack(test_data), os.path.join(self.data_dir, "test", f"{file_idx}.pt"))
            file_idx += 1

        # update the index_to_file dictionary
        self.index_to_file_dict["test"] = {
            i: os.path.join(f"{i}.pt") for i in range(file_idx)
        }

        # add the test_sample_size, num_sample_per_file_test and num_files to the index_to_file_dict
        self.index_to_file_dict["test"]["data_dir"] = os.path.join(self.data_dir, "test")
        self.index_to_file_dict["test"]["sample_size"] = self.test_sample_size
        self.index_to_file_dict["test"]["num_sample_per_file"] = self.num_sample_per_file_test
        self.index_to_file_dict["test"]["num_files"] = file_idx

        # add data_method.dict to the index_to_file_dict
        self.index_to_file_dict["data_method_hparams"] = self.data_method.dict

        with open(os.path.join(self.data_dir, "index_to_file_dict.yaml"), "w") as file:
            yaml.dump(self.index_to_file_dict, file)
        if return_index_to_file_dict:
            return self.index_to_file_dict
        
    def update_data_dir(self, new_data_dir: str):
        """
        This function is used to update the data directory.

        Parameters:
            data_dir (str): The new data directory.
        """
        with open(os.path.join(new_data_dir, "index_to_file_dict.yaml"), "r") as file:
            index_to_file_dict = yaml.safe_load(file)
        index_to_file_dict["data_dir"] = new_data_dir
        index_to_file_dict["train"]["data_dir"] = os.path.join(new_data_dir, "train")
        index_to_file_dict["val"]["data_dir"] = os.path.join(new_data_dir, "val")
        index_to_file_dict["test"]["data_dir"] = os.path.join(new_data_dir, "test")
        with open(os.path.join(new_data_dir, "index_to_file_dict.yaml"), "w") as file:
            yaml.dump(index_to_file_dict, file)
        self.data_dir = new_data_dir


class DatasetBase(Dataset):
    def __init__(self, 
                 index_to_file_dict: Dict = None,
                 data_method: DataMethod = None,
                 load_data_into_memory: bool = False, 
                 **data_method_args_dict
                 ):
        self.index_to_file_dict = index_to_file_dict
        # check if there are keywords "train", "val" and "test" in the index_to_file_dict.
        if "train" in self.index_to_file_dict.keys() or "val" in self.index_to_file_dict.keys() or "test" in self.index_to_file_dict.keys():
            raise ValueError("Please pass the specific index_to_file_dict for each of the train, val and test data.")
        
        self.data_dir = self.index_to_file_dict["data_dir"]
        
        self.data_method = data_method
        self.data_method_args_dict = data_method_args_dict if data_method_args_dict is not None else {}
        self.L = data_method_args_dict['data_method_args_dict']["L"] if "L" in self.data_method_args_dict['data_method_args_dict'] else 1

        self.load_data_into_memory = load_data_into_memory
        if self.load_data_into_memory:
            self.data = {}
            for file_idx in self.index_to_file_dict.keys():
                # file_idx may be str type, check type first
                if isinstance(file_idx, int):
                    self.data[file_idx] = torch.load(
                        os.path.join(self.data_dir, self.index_to_file_dict[file_idx]))

    def __len__(self):
        """
        This function is used to get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return self.index_to_file_dict["sample_size"]

    def __getitem__(self, idx):
        """
        This function is used to get the data from the given index.
        
        Parameters:
            idx (int): The index of the data.
        
        Returns:
            Any: The data.
        """
        # Determine which file and which sample within the file the index corresponds to
        file_index, sample_index = self._get_file_and_sample_index(idx)

        # Load the file only when it's needed
        data = torch.load(os.path.join(self.data_dir, self.index_to_file_dict[file_index])) if not self.load_data_into_memory else self.data[file_index]
        
        sample = self.data_method.__transform__(data[sample_index], zero_index=None, **self.data_method_args_dict)
        return sample

    def _get_file_and_sample_index(self, idx):
        """
        This function is used to get the file index and sample index from the given index.

        Parameters:
            idx (int): The index of the data.

        Returns:
            Tuple: The index of position to set y as 0, file index and sample index.
        """
        # Determine which position is needed to set y as 0
        # zero_index = idx // self.index_to_file_dict["sample_size"]

        # Set idx to number within file size
        # idx = idx % self.index_to_file_dict["sample_size"]

        # Determine which file the index corresponds to
        file_index = idx // self.index_to_file_dict["num_sample_per_file"]

        # Determine which sample within the file the index corresponds to
        sample_index = idx % self.index_to_file_dict["num_sample_per_file"]

        return file_index, sample_index



if __name__ == "__main__":
    from data_linear import LinearReg, LinearRegNormX
    from data_nonlinear import NonLinearSquareReg
    from args_parser import get_dataset_parser
    parser = get_dataset_parser()
    args = parser.parse_args()
    n_dim = args.n_dim
    n_pos = args.n_dim * 10
    train_size = args.train_size
    train_size = train_size * 64
    # train_size = train_size / n_pos
    train_size += 64
    train_size = int(train_size)
    print(train_size)
    # data_method = DataMethod({"seq_len": 100, "dim": 10, "noise_scale": 0.1})
    dataset_name = "testingnonorm" + str(n_dim) + "d"
    data_method = LinearReg({"L": n_pos, "dx": n_dim, "dy": 1, "number_of_samples": 1, "noise_std": 0.01})
    synthetic_data_generator = SyntheticDataGenerator(
            data_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)), dataset_name),
            train_sample_size=train_size,
            val_sample_size=train_size,
            test_sample_size=20,
            num_sample_per_file_train=2000 * 64,
            num_sample_per_file_val=2000 * 64,
            num_sample_per_file_test=20,
            data_method=data_method,
            data_method_args_dict={}
        )

    # Generate the data
    index_to_file_dict = synthetic_data_generator.generate_data(return_index_to_file_dict=True)

    # Create the dataset
    dataset = DatasetBase(
        index_to_file_dict=index_to_file_dict["train"], 
        data_method=data_method,
        data_method_args_dict={"L": n_pos})

    # Get the length of the dataset
    print(len(dataset))

    # Get the data from the given index
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    x = next(iter(train_dataloader))

    pass
