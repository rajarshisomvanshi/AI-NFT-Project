# TODO: Run AI-Project

## Step 1: Install and Run NFT Simulator
- [x] Install Streamlit if not already installed
- [x] Run the Streamlit app (nft_simulator.py)

## Step 2: Adapt and Run AI Model Training
- [x] Install PyTorch, torchvision, tqdm, and other dependencies (torch, torchvision, tqdm)
- [x] Create local datasets folder and update paths in notebook (change /kaggle/input/ and /kaggle/working/ to local paths like ./datasets/)
- [ ] Download or place Indian image datasets into ./datasets/ (e.g., indian-famous-personalities-image-dataset, etc.)
- [x] Convert ai-nft-project (1).ipynb to train_model.py script
- [ ] Run the training script to combine datasets, train model, and save checkpoints

## Step 3: Test Full Pipeline
- [ ] Generate sample images using the trained model
- [ ] Modify nft_simulator.py to allow uploading generated images or integrate generation
- [ ] Test minting NFTs with generated images in the simulator
