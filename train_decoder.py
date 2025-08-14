# train_decoder.py

import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


import config
from datasets.dataset import get_dataset
from model.scene_rep import JointEncoding 

class DecoderTrainer:
    """
    A class dedicated to offline training of the scene decoder.
    It encapsulates the model, optimizer, data loading, and training loop.
    Includes Early Stopping based on training loss to save the best model.
    """
    def __init__(self, cfg):
        """
        Initializes the trainer, model, optimizer, and data loader.
        """
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 1. Load Dataset
        print("Loading dataset...")
        self.train_dataset = get_dataset(cfg) 
        self.train_loader = DataLoader(self.train_dataset, batch_size=1, shuffle=True, num_workers=4)

        # 2. Initialize Model
        print("Initializing model...")
        self.bounding_box = torch.tensor(cfg['mapping']['bound']).float().to(self.device)
        self.model = JointEncoding(cfg, self.bounding_box).to(self.device)
        
        if self.cfg['multi_agents']['track_uncertainty']:
            voxel_size = self.cfg['NARUTO']['voxel_size']
            self.model.get_uncert_grid(voxel_size)
            print("Uncertainty grid initialized.")

        # 3. Create Optimizer
        print("Creating optimizer...")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg['training']['lr_decoder'])

        # 4. Early Stopping Parameters
        self.patience = self.cfg['training'].get('patience', 10)
        self.min_delta = self.cfg['training'].get('min_delta', 0.0001)
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.output_dir = os.path.join(self.cfg['data']['output'], self.cfg['data']['exp_name'])
        os.makedirs(self.output_dir, exist_ok=True)

    def get_loss_from_ret(self, ret):
        """
        Calculates the total loss from the model's output dictionary.
        """
        cfg_training = self.cfg['training']
        loss = 0
        loss += cfg_training['rgb_weight'] * ret['rgb_loss']
        loss += cfg_training['depth_weight'] * ret['depth_loss']
        loss += cfg_training['sdf_weight'] * ret["sdf_loss"]
        loss += cfg_training['fs_weight'] * ret["fs_loss"]
        
        if self.cfg["multi_agents"]["track_uncertainty"]:
            loss += cfg_training['uncert_weight'] * ret['uncert_loss']
            
        return loss

    def train_epoch(self, epoch_num):
        """
        Runs a single training epoch and returns the average loss.
        """
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_num+1}/{self.cfg['training']['epochs']}")
        
        for i, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            c2w = batch['c2w'][0].to(self.device)
            rays_d_cam = batch['direction'].squeeze(0).to(self.device)
            target_s = batch['rgb'].squeeze(0).to(self.device)
            target_d = batch['depth'].squeeze(0).to(self.device).unsqueeze(-1)
            H, W, _ = target_s.shape
            num_pixels = H * W
            sample_indices = torch.randint(0, num_pixels, (self.cfg['mapping']['sample'],))
            indice_h, indice_w = sample_indices // W, sample_indices % W
            rays_d_cam_sample = rays_d_cam[indice_h, indice_w, :]
            target_s_sample = target_s[indice_h, indice_w, :]
            target_d_sample = target_d[indice_h, indice_w, :]
            rays_o = c2w[:3, -1].expand(self.cfg['mapping']['sample'], -1)
            rays_d = torch.sum(rays_d_cam_sample[..., None, :] * c2w[:3, :3], -1)
            ret = self.model.forward(rays_o, rays_d, target_s_sample, target_d_sample)
            loss = self.get_loss_from_ret(ret)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        return total_loss / len(self.train_loader)

    def run(self):
        """
        The main entry point for the training process with early stopping.
        """
        print("Starting decoder training with early stopping based on training loss...")
        for epoch in range(self.cfg['training']['epochs']):
            avg_loss = self.train_epoch(epoch)
            
            print(f"Epoch {epoch+1}: Avg Train Loss: {avg_loss:.4f}, Best Loss: {self.best_loss:.4f}")

            # Check for improvement
            if self.best_loss - avg_loss > self.min_delta:
                print(f"Training loss improved from {self.best_loss:.4f} to {avg_loss:.4f}. Saving model...")
                self.best_loss = avg_loss
                self.patience_counter = 0
                self.save_best_decoder()
            else:
                self.patience_counter += 1
                print(f"Training loss did not improve significantly. Patience: {self.patience_counter}/{self.patience}")

            if self.patience_counter >= self.patience:
                print("Early stopping triggered as training loss has plateaued.")
                break
        
        print("Training loop finished.")

    def save_best_decoder(self):
        """
        Saves the best performing decoder's state_dict to a file.
        """
        decoder_state_dict = self.model.decoder.state_dict()
        save_path = os.path.join(self.output_dir, 'best_decoder.pt')
        save_dict = {'model': decoder_state_dict}
        torch.save(save_dict, save_path)
        print(f"Best decoder saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Offline training for the scene decoder with NARUTO modifications.')
    parser.add_argument('--config', type=str, required=True, help='Path to config file for training the decoder.')
    args = parser.parse_args()
    cfg = config.load_config(args.config)
    trainer = DecoderTrainer(cfg)
    trainer.run()