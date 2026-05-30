Model weights are saved here after training:

- best_action_model.pth         → best checkpoint (by val accuracy)
- action_recognition_final.pth  → final weights after all epochs

These files are excluded from GitHub (.gitignore) due to size (~45MB).
Share via Google Drive if needed.

To load manually:
    import torch, torch.nn as nn
    from torchvision.models.video import r3d_18

    model = r3d_18(pretrained=False)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 5))
    model.load_state_dict(torch.load('model/best_action_model.pth'))
    model.eval()
