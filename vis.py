import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

class InteractiveAttentionViewer:
    def __init__(
        self, 
        pt_path, 
        batch_idx=0,
        pixels_per_token=16
    ):
        self.pixels_per_token = pixels_per_token
        
        print("Loading data from checkpoint...")
        # Load the dictionary saved by the generation script
        data_dict = torch.load(pt_path, weights_only=True)
        
        # Read dimensions directly from the saved configuration
        self.target_width = data_dict["out_width"]
        self.target_height = data_dict["out_height"]
        
        self.h_tok = self.target_height // pixels_per_token
        self.w_tok = self.target_width // pixels_per_token
        self.total_tokens = self.h_tok * self.w_tok

        # Load Attention Matrix
        print("Processing attention tensor...")
        attn_matrix = data_dict["attention_map"][batch_idx]
        if torch.is_tensor(attn_matrix):
            self.attn_matrix = attn_matrix.detach().cpu().float().numpy()
        else:
            self.attn_matrix = attn_matrix
            
        expected_shape = (self.total_tokens, self.total_tokens)
        if self.attn_matrix.shape != expected_shape:
            raise ValueError(f"Shape mismatch! Expected {expected_shape}, got {self.attn_matrix.shape}")

        # Load Image from the saved tensor
        print("Restoring image from tensor...")
        if "image" in data_dict:
            # The saved tensor is (C, H, W) with values in [0.0, 1.0]
            img_tensor = data_dict["image"][batch_idx]
            # Matplotlib imshow expects (H, W, C)
            self.img_array = img_tensor.permute(1, 2, 0).numpy()
        else:
            raise KeyError("The key 'image' was not found in the checkpoint dictionary.")

        print("Data loaded successfully! Initializing UI...")
        self.setup_ui()

    def setup_ui(self):
        # Initialize the figure and axes
        self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 7 * (self.target_height / self.target_width)))
        self.fig.canvas.manager.set_window_title("FLUX.1-Fill ICL Attention Viewer")

        # --- Left Panel: Original Image ---
        self.axes[0].imshow(self.img_array)
        self.axes[0].set_title("Image (Click anywhere to select a token)")
        self.axes[0].axis('off')

        # Add visual aids: Crosshairs to separate the 2x2 ICL grid
        self.axes[0].axhline(self.target_height / 2, color='white', linestyle='--', alpha=0.5, linewidth=1)
        self.axes[0].axvline(self.target_width / 2, color='white', linestyle='--', alpha=0.5, linewidth=1)

        # Initialize the red bounding box (hidden at top-left initially)
        self.rect = patches.Rectangle(
            (0, 0), self.pixels_per_token, self.pixels_per_token,
            linewidth=2, edgecolor='red', facecolor='none', alpha=0.8
        )
        self.axes[0].add_patch(self.rect)

        # --- Right Panel: Attention Map ---
        # Use a zero matrix as a placeholder
        init_attn_map = np.zeros((self.h_tok, self.w_tok))
        self.im_attn = self.axes[1].imshow(init_attn_map, cmap='inferno')
        self.axes[1].set_title("Attention Map (Waiting for click...)")
        self.axes[1].axis('off')
        
        # Add colorbar
        self.cbar = self.fig.colorbar(
            self.im_attn, ax=self.axes[1], 
            fraction=0.046 * (self.target_height / self.target_width), pad=0.04
        )

        # Bind the mouse click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        plt.tight_layout()
        plt.show()

    def on_click(self, event):
        # Ensure the click event occurred within the left image panel
        if event.inaxes != self.axes[0]:
            return

        x_pixel, y_pixel = event.xdata, event.ydata
        if x_pixel is None or y_pixel is None:
            return

        # Calculate token grid coordinates
        x_tok = int(x_pixel // self.pixels_per_token)
        y_tok = int(y_pixel // self.pixels_per_token)

        # Clamp coordinates to valid ranges
        x_tok = max(0, min(x_tok, self.w_tok - 1))
        y_tok = max(0, min(y_tok, self.h_tok - 1))

        token_idx = y_tok * self.w_tok + x_tok

        # Update the position of the red bounding box
        self.rect.set_xy((x_tok * self.pixels_per_token, y_tok * self.pixels_per_token))

        # Extract the attention row and compute the log scale
        attn_map = self.attn_matrix[token_idx]
        attn_map_2d = attn_map.reshape((self.h_tok, self.w_tok))
        log_attn_map_2d = np.log(attn_map_2d + 1e-6)

        # Update the image data efficiently
        self.im_attn.set_data(log_attn_map_2d)
        
        # Dynamically adjust the colorbar scale for better contrast
        self.im_attn.set_clim(vmin=log_attn_map_2d.min(), vmax=log_attn_map_2d.max())

        # Update subplot titles to reflect the selected token
        self.axes[0].set_title(f"Image (Token Index: {token_idx})\nPos: x={x_tok}, y={y_tok}")
        self.axes[1].set_title(f"Attention Map for Token {token_idx}")

        # Trigger a UI redraw
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    # The viewer is now fully self-contained based on the dictionary file
    viewer = InteractiveAttentionViewer(
        pt_path="controller_attention_store.pt",
        batch_idx=0
    )