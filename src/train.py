import torch
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import os

from model import ResNet50Encoder,Decoder
from utils import adain,calc_content_loss,calc_style_loss
from Dataset import NeuralArtTransferDataset

def training_step(content_image,style_image,encoder,decoder,optimizer,style_weight=10.0):
    optimizer.zero_grad()
    with torch.no_grad():
        content_feats = encoder(content_image)
        style_feats = encoder(style_image)
    
    content_h4 = content_feats[-1]
    style_h4 = style_feats[-1]

    target_feats = adain(content_h4,style_h4)

    generated_image = decoder(target_feats)
    gen_feats = encoder(generated_image)
    gen_h4 = gen_feats[-1]

    loss_c = calc_content_loss(gen_h4,target_feats)

    loss_s = calc_style_loss(gen_feats,style_feats)

    total_loss = loss_c + style_weight * loss_s
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), loss_c.item(), loss_s.item()


def train(encoder,decoder,content_dir,style_dir,num_epochs=20,batch_size = 16,save_dir = "checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    decoder.to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=1e-4)

    transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()])

    dataset = NeuralArtTransferDataset(content_dir, style_dir, transform=transform)
    MAX_IMAGES = 1600 * 5
    subset_indices = torch.randperm(len(dataset))[:MAX_IMAGES].tolist()
    mini_dataset = torch.utils.data.Subset(dataset, subset_indices)
    dataloader = torch.utils.data.DataLoader(mini_dataset, batch_size=batch_size, shuffle=True,num_workers=8, drop_last=True)

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        loop = tqdm(dataloader, leave = True)
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        
        for batch_idx, (content_imgs,style_imgs) in enumerate(loop):
            content_imgs = content_imgs.to(device)
            style_imgs = style_imgs.to(device)

            total_loss, content_loss, style_loss = training_step(content_imgs,style_imgs,encoder,decoder,optimizer)

            loop.set_postfix(loss = total_loss, content_loss = content_loss, style_loss = style_loss)
        
        torch.save(decoder.state_dict(), os.path.join(save_dir, f"decoder_epoch_{epoch+1}.pth"))
        print(f"Saved checkpoint for epoch {epoch+1}")


def main():
    CONTENT_DIR = "./datasets/coco/train2017"
    STYLE_DIR = "./datasets/wikiart" 
    print("Initializing networks...")
    encoder = ResNet50Encoder()
    decoder = Decoder()
    
    print("Starting training loop...")
    train(
        encoder=encoder, 
        decoder=decoder, 
        content_dir=CONTENT_DIR, 
        style_dir=STYLE_DIR, 
        num_epochs=20,          
        batch_size=32,      
        save_dir="checkpoints"
    )

if __name__ == "__main__":
    main()