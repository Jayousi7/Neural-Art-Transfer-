import torch

DECODER_URL = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth"
OUTPUT_PATH = "pretrained_decoder.pth"


def download():
    print(f"Downloading pre-trained AdaIN decoder from {DECODER_URL} ...")
    state_dict = torch.hub.load_state_dict_from_url(DECODER_URL, map_location="cpu", file_name=OUTPUT_PATH)
    torch.save(state_dict, OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    download()
